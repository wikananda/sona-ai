import threading
import unittest

from sona_ai.services.recording_job_manager import RecordingJobManager


class RecordingJobManagerTest(unittest.TestCase):
    def test_cancel_removes_queued_job_before_it_runs(self):
        manager = RecordingJobManager(max_workers=1)
        blocker_started = threading.Event()
        release_blocker = threading.Event()
        queued_ran = threading.Event()

        def blocking_job(cancel_event=None):
            blocker_started.set()
            release_blocker.wait(timeout=2)

        def queued_job(cancel_event=None):
            queued_ran.set()

        try:
            manager.submit("recording-1", "job-1", blocking_job)
            self.assertTrue(blocker_started.wait(timeout=1))
            manager.submit("recording-2", "job-2", queued_job)

            self.assertTrue(manager.cancel("job-2"))
            release_blocker.set()
        finally:
            manager.shutdown(wait=True)

        self.assertFalse(queued_ran.is_set())

    def test_cancel_signals_running_job(self):
        manager = RecordingJobManager(max_workers=1)
        started = threading.Event()
        canceled = threading.Event()

        def cooperative_job(cancel_event=None):
            started.set()
            self.assertIsNotNone(cancel_event)
            if cancel_event.wait(timeout=2):
                canceled.set()

        try:
            manager.submit("recording-1", "job-1", cooperative_job)
            self.assertTrue(started.wait(timeout=1))
            self.assertTrue(manager.cancel("job-1"))
        finally:
            manager.shutdown(wait=True)

        self.assertTrue(canceled.is_set())

    def test_cancel_recording_signals_all_matching_jobs(self):
        manager = RecordingJobManager(max_workers=1)
        blocker_started = threading.Event()
        release_blocker = threading.Event()

        def blocking_job(cancel_event=None):
            blocker_started.set()
            release_blocker.wait(timeout=2)

        try:
            manager.submit("recording-1", "job-1", blocking_job)
            self.assertTrue(blocker_started.wait(timeout=1))
            manager.submit("recording-1", "job-2", lambda cancel_event=None: None)
            manager.submit("recording-2", "job-3", lambda cancel_event=None: None)

            self.assertEqual(manager.cancel_recording("recording-1"), 2)
            release_blocker.set()
        finally:
            manager.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
