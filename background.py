"""Durable bounded jobs with a single owner and no replay of paid requests."""

import asyncio
import gc
import time
import uuid

from filelock import FileLock, Timeout

ACTIVE = {"queued", "translating", "running", "generated", "sending"}
LABELS = {
    "queued": "排队中",
    "translating": "翻译中",
    "running": "生成中",
    "generated": "待发送",
    "sending": "发送中",
    "delivered": "已发送",
    "failed": "生成失败",
    "interrupted": "生成结果未确认",
    "send_failed": "发送失败",
    "send_unknown": "发送结果未确认",
    "cancelled": "已取消",
}


class BackgroundQueue:
    def __init__(self, store, config, execute):
        self.store, self.config, self.execute = store, config, execute
        self.tasks = {}
        self.semaphore = asyncio.Semaphore(max(1, int(config.get("max_concurrency", 2))))
        self.lock = None
        self.closed = False
        self.last_cleanup = 0

    def start(self):
        if self.closed:
            raise ValueError("生图服务正在重载, 请稍后再试")
        if self.lock is not None:
            return
        lock = FileLock(self.store.root / "worker.lock", thread_local=False)
        try:
            lock.acquire(timeout=0)
        except Timeout:
            # A retired plugin may leave an unreachable callback cycle until GC.
            # Collecting it never unlocks a live, reachable queue owner.
            gc.collect()
            try:
                lock.acquire(timeout=0)
            except Timeout:
                raise ValueError("已有生图后台运行, 请稍后再试") from None
        self.lock = lock
        for job in reversed(self.store.jobs()):
            state = job["state"]
            if state in {"queued", "translating", "generated"}:
                if state == "translating":
                    self.transition(job, "queued")
                self._spawn(job)
            elif state == "running":
                self.transition(job, "interrupted")
            elif state == "sending":
                self.transition(job, "send_unknown")

    def submit(self, payload):
        self.start()
        if time.time() - self.last_cleanup > 3600:
            self.store.prune_jobs(
                time.time() - max(1, int(self.config.get("history_days", 7))) * 86400, ACTIVE
            )
            self.last_cleanup = time.time()
        active = [j for j in self.store.jobs() if j["state"] in ACTIVE]
        if len(active) >= max(1, int(self.config.get("queue_limit", 10))):
            raise ValueError("生图队列已满, 本次未提交")
        mine = [j for j in active if j["owner"] == payload["owner"]]
        if len(mine) >= max(1, int(self.config.get("max_user_tasks", 2))):
            raise ValueError("你的待处理任务已达上限")
        recent = self.store.jobs(payload["owner"], limit=1)
        cooldown = max(0, float(self.config.get("user_cooldown", 0)))
        if recent and time.time() - recent[0]["created"] < cooldown:
            raise ValueError("生图冷却中, 请稍后再试")
        job = {**payload, "id": uuid.uuid4().hex[:12], "state": "queued", "created": time.time()}
        self.store.save_job(job)
        self._spawn(job)
        return job["id"]

    def _spawn(self, job):
        self.tasks[job["id"]] = asyncio.create_task(self._run(job), name="nai-" + job["id"])

    def transition(self, job, state, **values):
        job.update(values, state=state, updated=time.time())
        self.store.save_job(job)

    async def _run(self, job):
        try:
            async with self.semaphore:
                await self.execute(job, self)
        except asyncio.CancelledError:
            state = job["state"]
            if self.closed:
                target = {"running": "interrupted", "sending": "send_unknown"}.get(state, state)
            else:
                target = "send_unknown" if state == "sending" else "cancelled"
            self.transition(job, target)
            raise
        except Exception as exc:
            state = {"sending": "send_unknown", "running": "interrupted"}.get(
                job["state"], "failed"
            )
            # Keep remote response bodies and credentials out of the persistent ledger.
            self.transition(job, state, error=type(exc).__name__)
        finally:
            self.tasks.pop(job["id"], None)

    async def cancel(self, owner, task_id=""):
        selected = [
            j
            for j in self.store.jobs(owner)
            if j["id"] in self.tasks and (not task_id or j["id"] == task_id)
        ]
        tasks = [self.tasks[j["id"]] for j in selected]
        for job, task in zip(selected, tasks):
            # A task cancelled before its first step never enters its finally block.
            self.transition(job, "send_unknown" if job["state"] == "sending" else "cancelled")
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for job in selected:
            self.tasks.pop(job["id"], None)
        return len(tasks)

    async def close(self):
        self.closed = True
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks.clear()
        if self.lock:
            self.lock.release()
            self.lock = None
