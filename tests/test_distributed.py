import unittest
from distributed_gptq.distributed.coordinator import Coordinator
from distributed_gptq.distributed.worker import Worker

class TestDistributedFunctionality(unittest.TestCase):

    def setUp(self):
        self.coordinator = Coordinator()
        self.worker = Worker()

    def test_coordinator_initialization(self):
        self.assertIsNotNone(self.coordinator)
        self.assertEqual(self.coordinator.status, "initialized")

    def test_worker_initialization(self):
        self.assertIsNotNone(self.worker)
        self.assertEqual(self.worker.status, "idle")

    def test_coordinator_worker_communication(self):
        self.coordinator.add_worker(self.worker)
        self.assertIn(self.worker, self.coordinator.workers)

    def test_worker_task_assignment(self):
        self.worker.assign_task("test_task")
        self.assertEqual(self.worker.current_task, "test_task")

    def test_coordinator_task_distribution(self):
        self.coordinator.add_worker(self.worker)
        tasks = ["task1", "task2"]
        self.coordinator.distribute_tasks(tasks)
        self.assertEqual(self.worker.current_task, "task1")

if __name__ == '__main__':
    unittest.main()