class IDCounter:
    def __init__(self):
        self._count = 0

    def next_id(self):
        self._count += 1
        return self._count

    def clear_count(self):
        self._count = 0
