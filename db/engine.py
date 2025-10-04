from sqlmodel import create_engine


class Engine:
    @staticmethod
    def instance():
        if not hasattr(Engine, "engine"):
            Engine.engine = create_engine(
                "sqlite:///./database.db", connect_args={"check_same_thread": False}
            )
        return Engine.engine
