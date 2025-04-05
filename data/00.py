import os
import time
import logging
import schedule
import random
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from email.mime.text import MIMEText
import smtplib

# تأكد من تحميل بيانات vader
nltk.download('vader_lexicon')

# تحميل المتغيرات البيئية من ملف .env إن وجد
# يمكن استخدام مكتبة dotenv إذا رغبت ولكن هنا نعتمد على os.environ فقط
ACCOUNT_BALANCE = float(os.environ.get("ACCOUNT_BALANCE", 10000))
ALERT_EMAIL = os.environ.get("ALERT_EMAIL", "")
SMTP_SERVER = os.environ.get("SMTP_SERVER", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")

# إعداد سجل الأحداث
logging.basicConfig(
    filename='trading.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# دالة إرسال تنبيهات البريد الإلكتروني (اختيارية)
def send_email_alert(subject, message):
    try:
        if not all([SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_EMAIL]):
            logging.warning("إعدادات البريد الإلكتروني غير مكتملة.")
            return
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = SMTP_USER
        msg['To'] = ALERT_EMAIL
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, [ALERT_EMAIL], msg.as_string())
        server.quit()
        logging.info("تم إرسال تنبيه البريد الإلكتروني.")
    except Exception as e:
        logging.error(f"خطأ في إرسال تنبيه البريد الإلكتروني: {e}")

# ==================== نظام الامتثال الإسلامي ====================
class ShariaCompliance:
    # قائمة الرموز المقبولة
    HALAL_SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD"]
    
    @staticmethod
    def is_halal(symbol):
        return symbol in ShariaCompliance.HALAL_SYMBOLS

# ==================== التحليل الفني ====================
class TechnicalAnalyzer:
    @staticmethod
    def get_historical_data(symbol, period="1d", interval="1h"):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                logging.error(f"بيانات {symbol} فارغة.")
                return None
            df.reset_index(inplace=True)
            return df
        except Exception as e:
            logging.error(f"خطأ في جلب بيانات {symbol}: {e}")
            return None

    @staticmethod
    def compute_indicators(df):
        try:
            # حساب RSI
            rsi_indicator = RSIIndicator(close=df['Close'], window=14)
            df['rsi'] = rsi_indicator.rsi()
            # حساب MACD
            macd_indicator = MACD(close=df['Close'])
            df['macd'] = macd_indicator.macd()
            # حساب Bollinger Bands
            bb_indicator = BollingerBands(close=df['Close'], window=20, window_dev=2)
            df['bb_lower'] = bb_indicator.bollinger_lband()
            df['bb_upper'] = bb_indicator.bollinger_hband()
            return df
        except Exception as e:
            logging.error(f"خطأ في حساب المؤشرات الفنية: {e}")
            return df

    @staticmethod
    def detect_candlestick_patterns(df):
        patterns = {'bullish_engulfing': False, 'hammer': False}
        try:
            if len(df) < 2:
                return patterns
            # استخدام آخر شمعتين
            prev, last = df.iloc[-2], df.iloc[-1]
            # نمط الابتلاع الصعودي
            if (last['Close'] > last['Open'] and prev['Close'] < prev['Open'] and
                last['Open'] < prev['Close'] and last['Close'] > prev['Open']):
                patterns['bullish_engulfing'] = True
            # نمط المطرقة: جسم صغير وظل سفلي طويل
            body = abs(last['Close'] - last['Open'])
            lower_shadow = last['Open'] - last['Low'] if last['Open'] < last['Close'] else last['Close'] - last['Low']
            if body > 0 and lower_shadow > 2 * body:
                patterns['hammer'] = True
        except Exception as e:
            logging.error(f"خطأ في تحليل نمط الشموع: {e}")
        return patterns

# ==================== تحليل عمق السوق (محاكاة) ====================
class MarketDepthAnalyzer:
    @staticmethod
    def get_market_depth(symbol):
        # في حال توفر API حقيقي يمكن استبدال هذا الجزء
        # هنا نقوم بمحاكاة بيانات عمق السوق
        bid_volume = random.uniform(1000, 5000)
        ask_volume = random.uniform(1000, 5000)
        total = bid_volume + ask_volume
        liquidity_imbalance = abs(bid_volume - ask_volume) / total if total != 0 else 0
        return {
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'liquidity_imbalance': liquidity_imbalance
        }

# ==================== التحليل الأساسي ====================
class FundamentalAnalyzer:
    sid = SentimentIntensityAnalyzer()
    
    @staticmethod
    def get_news_sentiment(symbol):
        try:
            # استخدام yfinance لجلب الأخبار إن وجدت
            ticker = yf.Ticker(symbol)
            news_items = ticker.news
            if not news_items:
                logging.warning(f"لا توجد أخبار لرمز {symbol}.")
                return 0.0
            # تحليل أول خبر
            headline = news_items[0].get("title", "")
            sentiment = FundamentalAnalyzer.sid.polarity_scores(headline)["compound"]
            return sentiment
        except Exception as e:
            logging.error(f"خطأ في جلب أو تحليل الأخبار للرمز {symbol}: {e}")
            return 0.0

    @staticmethod
    def get_gold_price():
        try:
            gold = yf.Ticker("GC=F")
            df = gold.history(period="1d", interval="1h")
            if df.empty:
                logging.warning("بيانات الذهب غير متوفرة.")
                return None
            return df['Close'].iloc[-1]
        except Exception as e:
            logging.error(f"خطأ في جلب سعر الذهب: {e}")
            return None

# ==================== نموذج التعلم الآلي ====================
class HybridModel:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.is_trained = False

    def _prepare_data(self, df):
        try:
            # التأكد من حساب المؤشرات الفنية أولاً
            df = TechnicalAnalyzer.compute_indicators(df)
            df.dropna(inplace=True)
            # نستخدم الأعمدة: rsi، macd، bb_lower، bb_upper وحساب التغير النسبي للإغلاق
            df['pct_change'] = df['Close'].pct_change()
            df.dropna(inplace=True)
            features = df[['rsi', 'macd', 'bb_lower', 'bb_upper', 'pct_change']]
            # التسمية: 1 إذا ارتفع الإغلاق التالي، 0 خلاف ذلك
            labels = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)[:-1]
            features = features[:-1]
            return features, labels
        except Exception as e:
            logging.error(f"خطأ في تجهيز بيانات التدريب: {e}")
            return None, None

    def train(self, df):
        features, labels = self._prepare_data(df)
        if features is None or len(features) < 100:
            logging.warning("البيانات غير كافية لتدريب النموذج.")
            return
        try:
            self.model.fit(features, labels)
            self.is_trained = True
            logging.info("تم تدريب نموذج التعلم الآلي بنجاح.")
        except Exception as e:
            logging.error(f"خطأ أثناء تدريب النموذج: {e}")

    def predict(self, df):
        if not self.is_trained:
            logging.warning("النموذج غير مدرب، إرجاع قيمة افتراضية 0.5.")
            return 0.5
        try:
            df = TechnicalAnalyzer.compute_indicators(df)
            df.dropna(inplace=True)
            df['pct_change'] = df['Close'].pct_change()
            df.dropna(inplace=True)
            features = df[['rsi', 'macd', 'bb_lower', 'bb_upper', 'pct_change']]
            latest = features.iloc[-1].values.reshape(1, -1)
            prob = self.model.predict_proba(latest)[0][1]
            logging.info(f"توقع النموذج: {prob:.2f}")
            return prob
        except Exception as e:
            logging.error(f"خطأ أثناء التنبؤ: {e}")
            return 0.5

# ==================== إدارة المخاطر ====================
class RiskManager:
    MAX_RISK = 0.02  # 2% من الرصيد
    
    @staticmethod
    def calculate_position_size(balance, volatility):
        try:
            size = balance * RiskManager.MAX_RISK / volatility
            return min(size, 0.1)
        except Exception as e:
            logging.error(f"خطأ في حساب حجم الصفقة: {e}")
            return 0.01
    
    @staticmethod
    def check_drawdown(current_balance, peak_balance):
        if peak_balance == 0:
            return False
        drawdown = (peak_balance - current_balance) / peak_balance
        logging.info(f"التراجع الحالي: {drawdown:.2f}")
        return drawdown < 0.05

# ==================== النظام الرئيسي للتداول ====================
class IslamicTradingBot:
    def __init__(self):
        # التأكد من أن الرموز متوافقة مع معايير الشريعة
        self.symbols = [s for s in ShariaCompliance.HALAL_SYMBOLS]
        self.account_balance = ACCOUNT_BALANCE
        self.peak_balance = ACCOUNT_BALANCE
        self.model = HybridModel()
        logging.info("تم تهيئة النظام الرئيسي للتداول.")

    def analyze_market(self, symbol):
        if not ShariaCompliance.is_halal(symbol):
            logging.info(f"الرمز {symbol} غير متوافق مع المعايير الإسلامية.")
            return None
        
        df = TechnicalAnalyzer.get_historical_data(symbol, period="5d", interval="1h")
        if df is None or df.empty:
            logging.error(f"لا توجد بيانات للسوق للرمز {symbol}.")
            return None
        
        df = TechnicalAnalyzer.compute_indicators(df)
        patterns = TechnicalAnalyzer.detect_candlestick_patterns(df)
        market_depth = MarketDepthAnalyzer.get_market_depth(symbol)
        news_sentiment = FundamentalAnalyzer.get_news_sentiment(symbol)
        gold_price = FundamentalAnalyzer.get_gold_price()
        
        # تدريب النموذج باستخدام البيانات الحالية
        self.model.train(df)
        
        analysis = {
            'dataframe': df,
            'symbol': symbol,
            'price': df['Close'].iloc[-1],
            'rsi': df['rsi'].iloc[-1],
            'macd': df['macd'].iloc[-1],
            'bb_lower': df['bb_lower'].iloc[-1],
            'bb_upper': df['bb_upper'].iloc[-1],
            'patterns': patterns,
            'liquidity_ratio': market_depth['liquidity_imbalance'],
            'news_sentiment': news_sentiment,
            'gold_price': gold_price
        }
        logging.info(f"تم تحليل السوق للرمز {symbol} بنجاح.")
        return analysis

    def execute_trade(self, symbol):
        analysis = self.analyze_market(symbol)
        if analysis is None:
            logging.info(f"فشل تحليل السوق للرمز {symbol}.")
            return
        
        if not RiskManager.check_drawdown(self.account_balance, self.peak_balance):
            logging.warning("تجاوز التراجع المسموح به، إيقاف تنفيذ الصفقة.")
            return
        
        probability = self.model.predict(analysis['dataframe'])
        volatility = analysis['price'] / analysis['bb_lower'] if analysis['bb_lower'] != 0 else 1
        
        # شروط الدخول: مثال على شروط فنية أساسية
        if (analysis['patterns']['bullish_engulfing'] and
            analysis['rsi'] < 70 and
            analysis['macd'] > 0 and
            analysis['news_sentiment'] > 0.2 and
            probability > 0.75):
            
            lot = RiskManager.calculate_position_size(self.account_balance, volatility)
            # هنا نقوم بمحاكاة تنفيذ الصفقة (يمكن استبدالها بواجهة تداول حقيقية لاحقاً)
            logging.info(f"تنفيذ صفقة شراء للرمز {symbol} عند السعر {analysis['price']} بحجم {lot:.2f}")
            send_email_alert("تنبيه تنفيذ صفقة", f"تم تنفيذ صفقة شراء للرمز {symbol} عند السعر {analysis['price']}")
            # تحديث رصيد الحساب (محاكاة)
            profit = random.uniform(-0.01, 0.02) * self.account_balance
            self.account_balance += profit
            self.peak_balance = max(self.peak_balance, self.account_balance)
        else:
            logging.info(f"شروط الدخول غير متوافرة للرمز {symbol} (احتمالية: {probability:.2f}).")

# ==================== الحلقة الرئيسية ====================
def main():
    try:
        bot = IslamicTradingBot()
        # جدولة التنفيذ لكل رمز كل 30 دقيقة
        for symbol in bot.symbols:
            schedule.every(30).minutes.do(bot.execute_trade, symbol)
        logging.info("بدء الحلقة الرئيسية للتنفيذ...")
        while True:
            schedule.run_pending()
            time.sleep(1)
    except Exception as e:
        logging.critical(f"فشل النظام الرئيسي: {e}")
        send_email_alert("فشل النظام الرئيسي", str(e))

if __name__ == "__main__":
    main()
