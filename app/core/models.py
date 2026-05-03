from sqlalchemy import Column, Integer, String, Boolean, Float, ForeignKey, Date, DateTime, Text
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from .database import Base

class User(Base):
    __tablename__ = "users"
    
    username = Column(String, primary_key=True, index=True)
    full_name = Column(String, default="")
    phone = Column(String, default="")
    department = Column(String, default="")
    position = Column(String, default="")
    registered = Column(Boolean, default=True)
    is_doctorant = Column(Boolean, default=False)   # Doctorant talaba yoki yo'q
    staff_rate = Column(Float, default=1.0)          # Shtat birligi: 0.25, 0.5, 0.75, 1.0, 1.5, 2.0

    encodings = relationship("FaceEncoding", back_populates="user", cascade="all, delete-orphan")
    attendances = relationship("Attendance", back_populates="user", cascade="all, delete-orphan")

class FaceEncoding(Base):
    __tablename__ = "face_encodings"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, ForeignKey("users.username", ondelete="CASCADE"), index=True)
    embedding = Column(Vector(128))

    user = relationship("User", back_populates="encodings")

class Attendance(Base):
    __tablename__ = "attendances"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, ForeignKey("users.username", ondelete="CASCADE"), index=True)
    date = Column(String, index=True) # ISO format YYYY-MM-DD
    check_in_time = Column(String, nullable=True) # ISO format
    check_out_time = Column(String, nullable=True) # ISO format

    user = relationship("User", back_populates="attendances")

class Admin(Base):
    __tablename__ = "admins"
    
    username = Column(String, primary_key=True, index=True)
    password_hash = Column(String)
    role = Column(String, default="admin")

class Department(Base):
    __tablename__ = "departments"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, default="")

class Holiday(Base):
    __tablename__ = "holidays"
    
    date = Column(String, primary_key=True, index=True) # ISO format YYYY-MM-DD
    label = Column(String)

class AuditLogEntry(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, index=True) # ISO format
    admin_user = Column(String)
    action = Column(String)
    details = Column(Text)
