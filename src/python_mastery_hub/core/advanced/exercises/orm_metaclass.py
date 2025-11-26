"""ORM metaclass exercise - build ORM models using metaclasses."""


class ORMMetaclassExercise:
    """Build an ORM using Python metaclasses."""

    def __init__(self):
        self.name = "ORM Metaclass Exercise"
        self.description = "Create a lightweight ORM with metaclasses"

    def get_exercise(self):
        return {
            "title": "ORM Metaclass",
            "description": "Build a simple ORM framework using Python metaclasses and descriptors",
            "difficulty": "advanced",
        }


__all__ = ["ORMMetaclassExercise"]
