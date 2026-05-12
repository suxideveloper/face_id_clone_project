from app.core.database import SessionLocal
from app.core.models import User

# Person type konstantalari
PERSON_TYPE_STAFF = "staff"
PERSON_TYPE_DOCTORANT = "doctorant"
PERSON_TYPE_VISITOR = "visitor"
PERSON_TYPE_CONSULTANT = "consultant"
PERSON_TYPE_PROJECT_MEMBER = "project_member"

PERSON_TYPE_LABELS = {
    PERSON_TYPE_STAFF: "Asosiy xodim",
    PERSON_TYPE_DOCTORANT: "Doktorant",
    PERSON_TYPE_VISITOR: "Tashrif buyuruvchi",
    PERSON_TYPE_CONSULTANT: "Konsultant",
    PERSON_TYPE_PROJECT_MEMBER: "Loyiha ishtirokchisi",
}

def _user_to_dict(u: User) -> dict:
    """User modelini dict ga aylantiradi."""
    return {
        "full_name": u.full_name or "",
        "phone": u.phone or "",
        "department": u.department or "",
        "position": u.position or "",
        "registered": u.registered,
        "is_doctorant": (u.person_type == PERSON_TYPE_DOCTORANT) or (u.is_doctorant or False),
        "staff_rate": u.staff_rate if u.staff_rate is not None else 1.0,
        "person_type": u.person_type or PERSON_TYPE_STAFF,
        "notes": u.notes or "",
    }


class UserDatabase:
    def create_user(self, user_id: str, data: dict) -> bool:
        """Create a new user with profile data."""
        with SessionLocal() as db:
            if db.query(User).filter(User.username == user_id).first():
                return False
            person_type = data.get("person_type", PERSON_TYPE_STAFF)
            # person_type dan is_doctorant ni sinxronlashtirish (orqaga moslik)
            is_doc = (person_type == PERSON_TYPE_DOCTORANT) or data.get("is_doctorant", False)
            u = User(
                username=user_id,
                full_name=data.get("full_name", ""),
                phone=data.get("phone", ""),
                department=data.get("department", ""),
                position=data.get("position", ""),
                registered=True,
                is_doctorant=is_doc,
                staff_rate=data.get("staff_rate", 1.0),
                person_type=person_type,
                notes=data.get("notes", ""),
            )
            db.add(u)
            db.commit()
            return True

    def get_user(self, user_id: str) -> dict:
        """Get user data by ID (name)."""
        with SessionLocal() as db:
            u = db.query(User).filter(User.username == user_id).first()
            if u:
                return _user_to_dict(u)
            return None

    def get_all_users(self) -> dict:
        """Get all users."""
        with SessionLocal() as db:
            users = db.query(User).all()
            return {u.username: _user_to_dict(u) for u in users}

    def get_users_by_type(self, person_type: str) -> dict:
        """Generic: person_type bo'yicha foydalanuvchilarni qaytaradi.
        Kelajakda yangi kategoriya qo'shilganda faqat shu metoddan foydalaniladi.
        """
        with SessionLocal() as db:
            users = db.query(User).filter(User.person_type == person_type).all()
            return {u.username: _user_to_dict(u) for u in users}

    def get_staff_users(self) -> dict:
        """Faqat asosiy xodimlarni qaytaradi (person_type='staff').
        NULL person_type qiymati ham xodim sifatida hisoblanadi (orqaga moslik).
        """
        with SessionLocal() as db:
            users = db.query(User).filter(
                (User.person_type == PERSON_TYPE_STAFF) | (User.person_type == None)
            ).all()
            return {u.username: _user_to_dict(u) for u in users}

    def get_doctorant_users(self) -> dict:
        """Faqat doktorantlarni qaytaradi (person_type='doctorant')."""
        return self.get_users_by_type(PERSON_TYPE_DOCTORANT)

    def get_visitors(self) -> dict:
        """Faqat tashrif buyuruvchilarni qaytaradi (person_type='visitor')."""
        return self.get_users_by_type(PERSON_TYPE_VISITOR)

    def get_consultants(self) -> dict:
        """Faqat konsultantlarni qaytaradi (person_type='consultant')."""
        return self.get_users_by_type(PERSON_TYPE_CONSULTANT)

    def get_project_members(self) -> dict:
        """Faqat loyiha ishtirokchilarini qaytaradi (person_type='project_member')."""
        return self.get_users_by_type(PERSON_TYPE_PROJECT_MEMBER)

    def update_user(self, user_id: str, data: dict) -> bool:
        """Update user profile data."""
        with SessionLocal() as db:
            u = db.query(User).filter(User.username == user_id).first()
            if not u:
                return False
            for k, v in data.items():
                if hasattr(u, k):
                    setattr(u, k, v)
            # person_type va is_doctorant ni sinxronlashtirish
            if "person_type" in data:
                u.is_doctorant = (data["person_type"] == PERSON_TYPE_DOCTORANT)
            elif "is_doctorant" in data:
                if data["is_doctorant"] and not u.person_type:
                    u.person_type = PERSON_TYPE_DOCTORANT
            db.commit()
            return True

    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        with SessionLocal() as db:
            u = db.query(User).filter(User.username == user_id).first()
            if u:
                db.delete(u)
                db.commit()
                return True
            return False

    def rename_user(self, old_id: str, new_id: str) -> bool:
        """Rename a user (change their ID/name)."""
        with SessionLocal() as db:
            u = db.query(User).filter(User.username == old_id).first()
            if u:
                u.username = new_id
                db.commit()
                return True
            return False


user_db = UserDatabase()
