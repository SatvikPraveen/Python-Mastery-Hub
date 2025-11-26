"""Transaction manager exercise - implement transaction handling with context managers."""


class TransactionManagerExercise:
    """Implement a transaction manager using context managers."""

    def __init__(self):
        self.name = "Transaction Manager Exercise"
        self.description = "Build a robust transaction manager"

    def get_exercise(self):
        return {
            "title": "Transaction Manager",
            "description": "Create a context manager for database transactions with rollback support",
            "difficulty": "advanced",
        }


__all__ = ["TransactionManagerExercise"]
