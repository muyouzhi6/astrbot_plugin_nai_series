"""Durable queue tests that also run without an AstrBot installation."""

import asyncio
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astrbot_plugin_nai_series.background import ACTIVE, BackgroundQueue
from astrbot_plugin_nai_series.storage import Store


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name))
        self.gate = asyncio.Event()
        self.started = []

        async def execute(job, queue):
            self.started.append(job["id"])
            queue.transition(job, "running")
            await self.gate.wait()
            queue.transition(job, "delivered")

        self.queue = BackgroundQueue(
            self.store, {"max_concurrency": 2, "max_user_tasks": 4, "queue_limit": 3}, execute
        )

    async def asyncTearDown(self):
        await self.queue.close()
        self.temp.cleanup()

    async def test_parallel_limit_and_fifo(self):
        ids = [self.queue.submit({"owner": "one"}) for _ in range(3)]
        await asyncio.sleep(0)
        self.assertEqual(self.started, ids[:2])
        with self.assertRaisesRegex(ValueError, "队列已满"):
            self.queue.submit({"owner": "two"})
        self.gate.set()
        await asyncio.gather(*list(self.queue.tasks.values()))
        self.assertEqual(self.started, ids)
        self.assertTrue(all(j["state"] == "delivered" for j in self.store.jobs()))

    async def test_user_limit_and_cooldown(self):
        self.queue.config["max_user_tasks"] = 1
        self.queue.submit({"owner": "one"})
        with self.assertRaisesRegex(ValueError, "上限"):
            self.queue.submit({"owner": "one"})
        self.queue.config.update(max_user_tasks=4, user_cooldown=10)
        with self.assertRaisesRegex(ValueError, "冷却"):
            self.queue.submit({"owner": "one"})
        self.queue.submit({"owner": "two"})

    async def test_cancel_before_task_runs_releases_capacity(self):
        task_id = self.queue.submit({"owner": "one"})
        self.assertEqual(await self.queue.cancel("other", task_id), 0)
        self.assertEqual(await self.queue.cancel("one", task_id), 1)
        self.assertEqual(self.store.job(task_id, "one")["state"], "cancelled")
        self.assertFalse(self.queue.tasks)
        self.assertFalse(self.started)

    async def test_restart_keeps_unsent_but_never_replays_running(self):
        for index, state in enumerate(["queued", "translating", "generated", "running", "sending"]):
            self.store.save_job({"id": state, "owner": "one", "state": state, "created": index})
        self.queue.execute = AsyncMock()
        self.queue.start()
        await asyncio.gather(*list(self.queue.tasks.values()))
        ids = [call.args[0]["id"] for call in self.queue.execute.call_args_list]
        self.assertEqual(ids, ["queued", "translating", "generated"])
        self.assertEqual(self.store.job("running", "one")["state"], "interrupted")
        self.assertEqual(self.store.job("sending", "one")["state"], "send_unknown")

    async def test_live_owner_lock_prevents_duplicate_recovery(self):
        self.queue.start()
        other = BackgroundQueue(self.store, {}, AsyncMock())
        with self.assertRaises(ValueError):
            other.start()
        await self.queue.close()
        other.start()
        await other.close()

    async def test_retention_never_deletes_active_image(self):
        for state in ["delivered", "running"]:
            self.store.save_job({"id": state, "owner": "one", "state": state, "created": 0})
            self.store.image_path(state).write_bytes(b"image")
        self.store.prune_jobs(time.time(), ACTIVE)
        self.assertIsNone(self.store.job("delivered", "one"))
        self.assertFalse(self.store.image_path("delivered").exists())
        self.assertTrue(self.store.image_path("running").exists())
