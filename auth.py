#!/usr/bin/env python3

"""
Authentication Module for Algo Trading Platform

Features:
- User registration with validation
- Secure password hashing (bcrypt)
- JWT token-based authentication
- Session management
- Data stored in JSON file (can be upgraded to database)
"""

import json
import hashlib
import secrets
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict
from threading import RLock

# JWT-like token using HMAC (simple but secure for this use case)
import hmac
import base64

# Configuration
DATA_DIR = Path(__file__).resolve().parent / "data"
USERS_FILE = DATA_DIR / "users.json"
SECRET_KEY = secrets.token_hex(32)  # Generate on startup (regenerates on restart)
TOKEN_EXPIRY_HOURS = 24
SALT_ROUNDS = 100000  # For PBKDF2

_users_lock = RLock()


def _ensure_data_dir():
    """Ensure data directory exists."""
    DATA_DIR.mkdir(exist_ok=True)
    if not USERS_FILE.exists():
        USERS_FILE.write_text(json.dumps({"users": []}, indent=2))


def _load_users() -> Dict:
    """Load users from JSON file."""
    _ensure_data_dir()
    with _users_lock:
        try:
            return json.loads(USERS_FILE.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return {"users": []}


def _save_users(data: Dict):
    """Save users to JSON file."""
    _ensure_data_dir()
    with _users_lock:
        USERS_FILE.write_text(json.dumps(data, indent=2, default=str))


def _hash_password(password: str, salt: str = None) -> tuple:
    """
    Hash password using PBKDF2-HMAC-SHA256.
    Returns: (hashed_password, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        SALT_ROUNDS
    )
    return base64.b64encode(hashed).decode('utf-8'), salt


def _verify_password(password: str, hashed: str, salt: str) -> bool:
    """Verify password against hash."""
    new_hash, _ = _hash_password(password, salt)
    return hmac.compare_digest(new_hash, hashed)


def _generate_token(user_id: str, email: str) -> str:
    """
    Generate a secure authentication token.
    Format: base64(payload).signature
    """
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": (datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS)).isoformat(),
        "iat": datetime.utcnow().isoformat()
    }
    
    payload_json = json.dumps(payload)
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode()
    
    signature = hmac.new(
        SECRET_KEY.encode(),
        payload_b64.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return f"{payload_b64}.{signature}"


def _verify_token(token: str) -> Optional[Dict]:
    """
    Verify and decode authentication token.
    Returns payload if valid, None otherwise.
    """
    try:
        parts = token.split('.')
        if len(parts) != 2:
            return None
        
        payload_b64, signature = parts
        
        # Verify signature
        expected_sig = hmac.new(
            SECRET_KEY.encode(),
            payload_b64.encode(),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(signature, expected_sig):
            return None
        
        # Decode payload
        payload_json = base64.urlsafe_b64decode(payload_b64.encode()).decode()
        payload = json.loads(payload_json)
        
        # Check expiry
        exp = datetime.fromisoformat(payload['exp'])
        if datetime.utcnow() > exp:
            return None
        
        return payload
    except Exception:
        return None


def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_password(password: str) -> tuple:
    """
    Validate password strength.
    Returns: (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    
    return True, ""


def validate_phone(phone: str) -> bool:
    """Validate phone number format."""
    # Remove spaces and dashes
    clean = re.sub(r'[\s\-\+]', '', phone)
    return len(clean) >= 10 and clean.isdigit()


def register_user(
    first_name: str,
    last_name: str,
    email: str,
    phone: str,
    password: str
) -> tuple:
    """
    Register a new user.
    Returns: (success, message, user_data)
    """
    # Validate inputs
    first_name = first_name.strip()
    last_name = last_name.strip()
    email = email.strip().lower()
    phone = phone.strip()
    
    if len(first_name) < 2:
        return False, "First name must be at least 2 characters", None
    
    if len(last_name) < 2:
        return False, "Last name must be at least 2 characters", None
    
    if not validate_email(email):
        return False, "Invalid email format", None
    
    if not validate_phone(phone):
        return False, "Invalid phone number", None
    
    is_valid_pwd, pwd_error = validate_password(password)
    if not is_valid_pwd:
        return False, pwd_error, None
    
    # Check if email already exists
    data = _load_users()
    for user in data["users"]:
        if user["email"] == email:
            return False, "Email already registered", None
    
    # Hash password
    hashed_password, salt = _hash_password(password)
    
    # Create user
    user_id = secrets.token_hex(16)
    user = {
        "id": user_id,
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone": phone,
        "password_hash": hashed_password,
        "password_salt": salt,
        "created_at": datetime.utcnow().isoformat(),
        "last_login": None,
        "is_active": True
    }
    
    data["users"].append(user)
    _save_users(data)
    
    # Return user data without sensitive info
    safe_user = {k: v for k, v in user.items() if k not in ("password_hash", "password_salt")}
    return True, "Registration successful", safe_user


def login_user(email: str, password: str) -> tuple:
    """
    Authenticate user and generate token.
    Returns: (success, message, {token, user_data})
    """
    email = email.strip().lower()
    
    if not email or not password:
        return False, "Email and password required", None
    
    data = _load_users()
    
    for user in data["users"]:
        if user["email"] == email:
            # Verify password
            if _verify_password(password, user["password_hash"], user["password_salt"]):
                if not user.get("is_active", True):
                    return False, "Account is deactivated", None
                
                # Update last login
                user["last_login"] = datetime.utcnow().isoformat()
                _save_users(data)
                
                # Generate token
                token = _generate_token(user["id"], user["email"])
                
                return True, "Login successful", {
                    "token": token,
                    "name": f"{user['first_name']} {user['last_name']}",
                    "email": user["email"],
                    "user_id": user["id"]
                }
            else:
                return False, "Invalid password", None
    
    return False, "Email not found", None


def verify_auth_token(token: str) -> tuple:
    """
    Verify authentication token.
    Returns: (is_valid, user_data)
    """
    if not token:
        return False, None
    
    # Remove "Bearer " prefix if present
    if token.startswith("Bearer "):
        token = token[7:]
    
    payload = _verify_token(token)
    if not payload:
        return False, None
    
    # Get user from database
    data = _load_users()
    for user in data["users"]:
        if user["id"] == payload["user_id"]:
            safe_user = {
                "id": user["id"],
                "name": f"{user['first_name']} {user['last_name']}",
                "email": user["email"]
            }
            return True, safe_user
    
    return False, None


def get_user_by_id(user_id: str) -> Optional[Dict]:
    """Get user by ID (without sensitive data)."""
    data = _load_users()
    for user in data["users"]:
        if user["id"] == user_id:
            return {k: v for k, v in user.items() if k not in ("password_hash", "password_salt")}
    return None


def get_all_users() -> list:
    """Get all users (admin function, without passwords)."""
    data = _load_users()
    return [
        {k: v for k, v in user.items() if k not in ("password_hash", "password_salt")}
        for user in data["users"]
    ]


# Initialize on import
_ensure_data_dir()
