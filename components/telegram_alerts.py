import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from helpers import format_alert_message

class TelegramAlerts:
    def __init__(self):
        self.base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
        
    def send_alert(self, message, parse_mode='Markdown'):
        """
        إرسال رسالة تنبيه إلى قناة/مجموعة Telegram
        
        :param message: نص الرسالة
        :param parse_mode: تنسيق الرسالة (Markdown أو HTML)
        :return: True إذا نجح الإرسال، False إذا فشل
        """
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            print("⚠️ Telegram credentials not configured!")
            return False
            
        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': parse_mode
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to send Telegram alert: {e}")
            return False

    def send_stock_alert(self, symbol, price, change, threshold):
        """
        إرسال تنبيه خاص بحركة سهم معين
        
        :param symbol: رمز السهم
        :param price: السعر الحالي
        :param change: نسبة التغيير
        :param threshold: الحد الذي تم تجاوزه
        """
        message = format_alert_message(symbol, price, change, threshold)
        return self.send_alert(message)

# مثال للاستخدام
if __name__ == "__main__":
    # اختبار إرسال تنبيه
    telegram = TelegramAlerts()
    
    # إرسال رسالة اختبارية
    telegram.send_alert("🚀 *تنبيه سوق الأسهم* \nهذه رسالة اختبار من نظام المراقبة")
    
    # إرسال تنبيه سهم (سيتطلب وجود دالة format_alert_message في helpers.py)
    # telegram.send_stock_alert("AAPL", 182.3, 5.2, 5)
