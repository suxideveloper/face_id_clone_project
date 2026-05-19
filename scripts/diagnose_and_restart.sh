#!/bin/bash
# ============================================================
# FaceID Server — Diagnostika va Restart Skripti
# SSH orqali servergga ulanib, shu faylni ishga tushiring:
#   bash diagnose_and_restart.sh
# ============================================================

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

echo -e "${CYAN}============================================${RESET}"
echo -e "${CYAN}  FaceID Server — Diagnostika${RESET}"
echo -e "${CYAN}============================================${RESET}"

# ── 1. Service nomi topish ────────────────────────────────
echo -e "\n${YELLOW}1. Mavjud serviclar tekshirilmoqda...${RESET}"
SERVICE_NAME=""

for name in faceid faceid-app face-id attendance uvicorn; do
    if systemctl list-units --type=service 2>/dev/null | grep -q "$name"; then
        SERVICE_NAME="$name"
        echo -e "${GREEN}   ✅ Service topildi: $SERVICE_NAME${RESET}"
        break
    fi
done

if [ -z "$SERVICE_NAME" ]; then
    echo -e "${YELLOW}   ⚠️  Systemd service topilmadi. Supervisor/PM2 tekshirilmoqda...${RESET}"
    
    # Supervisor tekshirish
    if command -v supervisorctl &>/dev/null; then
        echo -e "${CYAN}   Supervisor jarayonlari:${RESET}"
        supervisorctl status 2>/dev/null
    fi

    # PM2 tekshirish
    if command -v pm2 &>/dev/null; then
        echo -e "${CYAN}   PM2 jarayonlari:${RESET}"
        pm2 list 2>/dev/null
    fi

    # To'g'ridan-to'g'ri ishlab turgan uvicorn/python jarayonlari
    echo -e "\n${CYAN}   Ishlab turgan Python jarayonlari:${RESET}"
    ps aux | grep -E "python|uvicorn|main.py" | grep -v grep
fi

# ── 2. Port 8080 kimga tegishli ───────────────────────────
echo -e "\n${YELLOW}2. Port 8080 holati:${RESET}"
if ss -tlnp | grep -q ":8080"; then
    echo -e "${GREEN}   ✅ Port 8080 ochiq${RESET}"
    ss -tlnp | grep ":8080"
else
    echo -e "${RED}   ❌ Port 8080 yopiq — server ishlamayapti!${RESET}"
fi

# ── 3. Liveness fayl serverda bormi ─────────────────────
echo -e "\n${YELLOW}3. Yangi liveness fayllari tekshirilmoqda...${RESET}"

# Loyiha papkasini topish
PROJECT_DIR=""
for dir in /home/*/faceid /opt/faceid /var/www/faceid ~/faceid /root/faceid; do
    if [ -f "$dir/main.py" ]; then
        PROJECT_DIR="$dir"
        break
    fi
done

if [ -n "$PROJECT_DIR" ]; then
    echo -e "${GREEN}   Loyiha papkasi: $PROJECT_DIR${RESET}"
    
    if [ -f "$PROJECT_DIR/app/services/liveness.py" ]; then
        echo -e "${GREEN}   ✅ liveness.py mavjud${RESET}"
        echo -e "      Yaratilgan: $(stat -c '%y' $PROJECT_DIR/app/services/liveness.py | cut -d'.' -f1)"
    else
        echo -e "${RED}   ❌ liveness.py YO'Q — fayl ko'chirilmagan!${RESET}"
    fi

    # tracker.py yangilanganini tekshirish
    if grep -q "no_landmark_streak" "$PROJECT_DIR/app/services/tracker.py" 2>/dev/null; then
        echo -e "${GREEN}   ✅ tracker.py yangilangan (glasses mode bor)${RESET}"
    else
        echo -e "${RED}   ❌ tracker.py eski versiya — yangilanmagan!${RESET}"
    fi

    # routes.py yangilanganini tekshirish
    if grep -q "liveness_detector" "$PROJECT_DIR/app/api/routes.py" 2>/dev/null; then
        echo -e "${GREEN}   ✅ routes.py yangilangan${RESET}"
    else
        echo -e "${RED}   ❌ routes.py eski versiya!${RESET}"
    fi
else
    echo -e "${RED}   ❌ Loyiha papkasi topilmadi!${RESET}"
    echo -e "   main.py qayerda ekanini tekshiring: find / -name 'main.py' 2>/dev/null | grep -v venv"
fi

# ── 4. Service restart ─────────────────────────────────
echo -e "\n${YELLOW}4. Service restart qilish...${RESET}"

if [ -n "$SERVICE_NAME" ]; then
    echo -e "   sudo systemctl restart $SERVICE_NAME"
    sudo systemctl restart "$SERVICE_NAME"
    sleep 3
    
    STATUS=$(systemctl is-active "$SERVICE_NAME")
    if [ "$STATUS" = "active" ]; then
        echo -e "${GREEN}   ✅ Service muvaffaqiyatli restart bo'ldi!${RESET}"
    else
        echo -e "${RED}   ❌ Service ishga tushmadi! Status: $STATUS${RESET}"
        echo -e "\n${YELLOW}   Xato loglari:${RESET}"
        sudo journalctl -u "$SERVICE_NAME" -n 30 --no-pager
    fi
elif [ -n "$PROJECT_DIR" ]; then
    echo -e "${YELLOW}   Systemd service topilmadi. Qo'lda restart...${RESET}"
    
    # Eski jarayon o'ldirish
    pkill -f "uvicorn main:app" 2>/dev/null || true
    pkill -f "python main.py" 2>/dev/null || true
    sleep 2
    
    echo -e "${CYAN}   Server qayta ishga tushirilmoqda...${RESET}"
    cd "$PROJECT_DIR"
    
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
    
    nohup python main.py > /tmp/faceid.log 2>&1 &
    sleep 3
    
    if ss -tlnp | grep -q ":8080"; then
        echo -e "${GREEN}   ✅ Server ishga tushdi! Port 8080 ochiq${RESET}"
    else
        echo -e "${RED}   ❌ Server ishga tushmadi! Log:${RESET}"
        tail -20 /tmp/faceid.log
    fi
fi

# ── 5. Yakuniy holat ───────────────────────────────────
echo -e "\n${CYAN}============================================${RESET}"
echo -e "${CYAN}  5. Yakuniy holat tekshiruvi${RESET}"
echo -e "${CYAN}============================================${RESET}"

sleep 2
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/login 2>/dev/null | grep -q "200"; then
    echo -e "${GREEN}✅ Server javob bermoqda: http://localhost:8080${RESET}"
else
    echo -e "${RED}❌ Server javob bermayapti!${RESET}"
fi

if [ -n "$SERVICE_NAME" ]; then
    echo -e "\n${CYAN}Service holati:${RESET}"
    systemctl status "$SERVICE_NAME" --no-pager -l | head -20
fi

echo -e "\n${CYAN}Tayyor! Muammo davom etsa loglarni ko'ring:${RESET}"
echo -e "  sudo journalctl -u ${SERVICE_NAME:-faceid} -f"
echo -e "  tail -f /tmp/faceid.log"
