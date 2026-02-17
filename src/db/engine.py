from src.db.session import engine


class Engine:
    @staticmethod
    def instance():
        """Return the engine instance used by get_db() for compatibility."""
        return engine
