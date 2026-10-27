from functools import wraps

from app.vectorstore.write_lock import exclusive_write_lock


def maintenance_guard(vectorstore, timeout_seconds=0):
    return exclusive_write_lock(vectorstore.maintenance_lock_path, timeout_seconds)


def knowledge_mutation(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with maintenance_guard(self._vectorstore, timeout_seconds=600):
            return method(self, *args, **kwargs)

    return guarded
