import uuid
from app.core.database import SessionLocal
from app.core.models import Department


class DepartmentsDatabase:
    def get_all(self):
        with SessionLocal() as db:
            rows = db.query(Department).all()
            return [{"id": d.id, "name": d.name, "description": d.description} for d in rows]

    def get_by_id(self, dept_id: str):
        with SessionLocal() as db:
            d = db.query(Department).filter(Department.id == dept_id).first()
            if d:
                return {"id": d.id, "name": d.name, "description": d.description}
            return None

    def create(self, name: str) -> dict:
        new_id = str(uuid.uuid4())
        with SessionLocal() as db:
            d = Department(id=new_id, name=name.strip())
            db.add(d)
            db.commit()
            return {"id": d.id, "name": d.name}

    def update(self, dept_id: str, new_name: str) -> bool:
        with SessionLocal() as db:
            d = db.query(Department).filter(Department.id == dept_id).first()
            if d:
                d.name = new_name.strip()
                db.commit()
                return True
            return False

    def delete(self, dept_id: str) -> bool:
        with SessionLocal() as db:
            d = db.query(Department).filter(Department.id == dept_id).first()
            if d:
                db.delete(d)
                db.commit()
                return True
            return False


departments_db = DepartmentsDatabase()
