import time
import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
from vnstock import Vnstock, register_user
from datetime import datetime, timedelta
import warnings
import calendar

warnings.filterwarnings('ignore')

# ====================== VNSTOCK API KEY ======================
VNSTOCK_API_KEY = "vnstock_9008899a9dce77c13e296b6442ee866c"
try:
    register_user(api_key=VNSTOCK_API_KEY)
    st.success("✅ Đã đăng ký vnstock API key", icon="🔑")
except Exception as e:
    st.warning(f"⚠️ Không đăng ký được key: {e}")

st.set_page_config(page_title="Multi-View Trade Analyzer", layout="wide")
st.title("📊 Multi-View Trade - Phân tích cổ phiếu ngắn hạn")
st.markdown("**Kế hoạch trade 5-7 ngày | Target +4% | Stop -3% | Ngân sách 30 triệu**")

# ====================== DICTIONARY NGÀNH NGHỀ ======================
SECTOR_MAP = { ... }  # Giữ nguyên dictionary SECTOR_MAP của bạn (quá dài nên tao bỏ qua phần này, bạn copy lại từ file cũ)

def get_sector(symbol):
    return SECTOR_MAP.get(symbol, "Khác / Chưa phân loại")

# ====================== TRỌNG SỐ (ĐÃ THÊM ICHIMOKU) ======================
WEIGHTS = {
    'Momentum': 0.28,
    'Trend': 0.20,
    'Volume': 0.18,
    'Oscillator': 0.14,      # RSI + Stochastic
    'Volatility': 0.07,
    'PriceAction': 0.06,
    'Ichimoku': 0.07         # View mới
}

# ====================== HÀM CHẤM ĐIỂM ĐÃ TỐI ƯU ======================
def score_momentum(crsi):
    if crsi > 68: return 9.5
    elif crsi > 58: return 8.2
    elif crsi > 52: return 7.0
    elif crsi < 45: return 3.8
    else: return 5.5

def score_trend(price, ma20, ma50):
    if price > ma20 > ma50: return 9.2
    elif ma20 > ma50: return 7.0
    elif price > ma20: return 5.5
    else: return 3.5

def score_oscillator(rsi, stoch):
    if 48 <= rsi <= 68 and stoch > 55: return 9.0
    elif 40 <= rsi <= 72: return 7.0
    elif rsi > 72 or rsi < 35: return 4.0
    else: return 5.5

def score_volume(obv_trend, vol_ratio):
    if obv_trend == "up" and vol_ratio > 1.5: return 9.5
    elif obv_trend == "up" and vol_ratio > 1.25: return 8.0
    elif obv_trend == "up": return 6.8
    elif obv_trend == "flat" and vol_ratio > 1.2: return 6.0
    elif obv_trend == "down": return 3.8
    elif vol_ratio < 0.85: return 4.0
    else: return 5.5

def score_volatility(band_width):
    if band_width < 0.08: return 8.5
    elif band_width > 0.18: return 4.5
    else: return 6.5

def score_price_action(current_price, support, df):
    if len(df) < 5: return 5.0
    distance = (current_price - support) / support
    if distance > 0.018 and df['low'].iloc[-1] <= support * 1.008:
        return 9.2
    elif 0.005 < distance <= 0.018:
        return 7.8
    elif distance <= 0.005:
        return 3.8 if df['close'].iloc[-1] < df['open'].iloc[-1] else 6.0
    else:
        return 4.5

def score_ichimoku(price, cloud_top, cloud_bottom, chikou, tenkan, kijun, rsi):
    """Kết hợp Ichimoku với RSI để tăng độ chính xác"""
    # Bullish mạnh: Giá trên đám mây + Chikou confirm + RSI không quá mua
    if price > cloud_top and chikou > price and rsi < 72:
        return 9.3
    # Bullish: Giá trên đám mây
    elif price > cloud_top:
        return 7.8
    # Bearish mạnh: Giá dưới đám mây + Chikou confirm
    elif price < cloud_bottom and chikou < price:
        return 3.5
    # Giá trong đám mây (sideway)
    else:
        return 5.2 if 45 <= rsi <= 65 else 4.0

# ====================== CALCULATE VIEW SCORES ======================
def calculate_view_scores(df, current_price, support):
    scores = {}
    try:
        # Các chỉ báo cơ bản
        ma20 = df['close'].rolling(20).mean()
        ma50 = df['close'].rolling(50).mean()
        rsi = ta.rsi(df['close'], length=14).iloc[-1] if len(df) > 14 else 50.0
        stoch_df = ta.stoch(df['high'], df['low'], df['close'])
        stoch_k = stoch_df['STOCHk_14_3_3'].iloc[-1] if not stoch_df.empty else 50.0
        
        obv = ta.obv(df['close'], df['volume'])
        obv_trend = "up" if obv.diff().iloc[-1] > 0 else "down" if obv.diff().iloc[-1] < 0 else "flat"
        vol_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1] if len(df) > 20 else 1.0
        
        crsi = ta.crsi(df['close'], df['high'], df['low'], length=3, fast=2, slow=100).iloc[-1] if len(df) > 100 else 50.0
        
        bb = ta.bbands(df['close'], length=20, std=2)
        band_width = (bb['BBU_20_2.0'].iloc[-1] - bb['BBL_20_2.0'].iloc[-1]) / current_price if not bb.empty else 0.12

        # ====================== ICHIMOKU ======================
        ichi = ta.ichimoku(df['high'], df['low'], df['close'])
        cloud_top = ichi[0]['ISA_9'].iloc[-1]
        cloud_bottom = ichi[0]['ISB_26'].iloc[-1]
        chikou = ichi[1]['chikou'].iloc[-1] if 'chikou' in ichi[1].columns else df['close'].iloc[-1]
        tenkan = ichi[0]['TENKAN_9'].iloc[-1]
        kijun = ichi[0]['KIJUN_26'].iloc[-1]

    except Exception as e:
        st.warning(f"Lỗi tính indicator cho {symbol}: {str(e)[:80]}")
        rsi = stoch_k = crsi = 50.0
        obv_trend = "flat"
        vol_ratio = 1.0
        band_width = 0.12
        cloud_top = cloud_bottom = current_price
        chikou = tenkan = kijun = current_price

    scores['Momentum']   = score_momentum(crsi)
    scores['Trend']      = score_trend(current_price, ma20.iloc[-1], ma50.iloc[-1])
    scores['Oscillator'] = score_oscillator(rsi, stoch_k)
    scores['Volume']     = score_volume(obv_trend, vol_ratio)
    scores['Volatility'] = score_volatility(band_width)
    scores['PriceAction']= score_price_action(current_price, support, df)
    scores['Ichimoku']   = score_ichimoku(current_price, cloud_top, cloud_bottom, chikou, tenkan, kijun, rsi)

    return scores

# ====================== WEIGHTED SCORE ======================
def calculate_weighted_score(scores_dict):
    weighted = sum(scores_dict.get(view, 5.0) * weight for view, weight in WEIGHTS.items())
    
    strong = sum(1 for v in scores_dict.values() if v >= 7.5)
    if strong >= 5: weighted += 1.2
    elif strong >= 4: weighted += 0.8
    
    # Synergy Momentum + Ichimoku (rất mạnh khi cả hai cùng bullish)
    if scores_dict.get('Momentum', 0) >= 8.0 and scores_dict.get('Ichimoku', 0) >= 8.0:
        weighted += 1.0
    
    weak = sum(1 for v in scores_dict.values() if v <= 4.0)
    if weak >= 3: weighted -= 0.8

    return round(min(max(weighted, 3.0), 10.5), 2)

# ====================== FIBONACCI & MARKET CONTEXT (giữ nguyên) ======================
def calculate_fibonacci(df):
    high = df['high'].rolling(60, min_periods=20).max().iloc[-1]
    low = df['low'].rolling(60, min_periods=20).min().iloc[-1]
    diff = high - low
    return round(high - diff * 0.382, 2), round(high - diff * 0.5, 2), round(high - diff * 0.618, 2)

def get_market_context():
    today = datetime.now().date()
    weekday = today.weekday()
    day_factor = 0.0
    day_note = ""

    if weekday == 0:
        day_factor = -0.8
        day_note = "Đầu tuần - Dễ bị đạp giá"
    elif weekday in [2, 3]:
        day_factor = 0.6
        day_note = "Giữa tuần - Dễ tạo sóng đẩy"
    elif weekday == 4:
        day_factor = 0.4
        day_note = "Cuối tuần - Hay thoát hàng"

    is_near_exp = False
    days_to_exp = 0
    exp_text = "—"

    year, month = today.year, today.month
    last_day = calendar.monthrange(year, month)[1]
    for d in range(max(1, last_day - 9), last_day + 1):
        try:
            check_date = datetime(year, month, d).date()
            if check_date.weekday() in [2, 3]:
                days_diff = (check_date - today).days
                if -3 <= days_diff <= 3:
                    is_near_exp = True
                    days_to_exp = days_diff
                    if days_diff == 0:
                        exp_text = "Hôm nay đáo hạn"
                    elif days_diff > 0:
                        exp_text = f"Còn {days_diff} ngày"
                    else:
                        exp_text = f"Đã qua {-days_diff} ngày"
                    break
        except:
            continue

    return day_factor, day_note, is_near_exp, days_to_exp, exp_text

# ====================== GIAO DIỆN & LOGIC CHẠY ======================
st.sidebar.header("⚙️ Cài đặt phân tích")

manual_stocks = ["HPG", "FPT", "TCB", "SSI", "STB", "VND", "MWG", "MBB", "VHM", "VIC", "VPB", "DIG", "NVL", "GEX", "VCI", "MSN", "VNM", "ACB", "CTG", "SHB", "HDB", "VIX", "KBC", "PDR", "DXG", "VCB", "DGC", "TPB", "HSG", "NKG", "VRE", "EIB", "POW", "GAS", "LPB", "TCH", "VJC", "BID", "PLX", "SAB", "BVH", "REE", "PNJ", "GVR", "FRT", "FTS", "CTS", "BSI", "VHC", "ANV", "IDC", "KDH", "NLG", "DBC", "PVS", "PVD", "SCS", "VOS", "PVT", "HAH", "DCM", "DPM", "PC1", "GEG", "VGT", "TNG", "MSB", "OCB", "VIB", "BAB", "TTA", "BCG", "HDG", "SAM", "AAA", "PHR", "SZC", "VPI", "CII", "HHV", "LCG", "VCG", "LSS", "SBT", "QNS", "MIG", "GIL", "VNA", "SKG", "VSC", "BWE", "TDM", "NT2", "PET", "DGW", "CSV", "LAS", "BFC", "VFG", "VPH"]

selection_mode = st.sidebar.radio(
    "Chế độ chọn cổ phiếu",
    options=["Chọn thủ công", "Chọn 30 cổ phiếu VN30", "Chọn 100 cổ phiếu thanh khoản lớn nhất"],
    horizontal=True
)

if selection_mode == "Chọn thủ công":
    selected_stocks = st.sidebar.multiselect(
        "Chọn cổ phiếu", 
        options=manual_stocks, 
        default=["ACB", "VCB", "TCB", "HPG", "FPT", "MWG", "SSI"]
    )
elif selection_mode == "Chọn 30 cổ phiếu VN30":
    vn30_list = ["ACB", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG", "MBB","MSN", "MWG", "NVL", "PNJ", "POW", "SAB", "SSI", "STB", "TCB", "TPB","VCB", "VHM", "VIC", "VJC", "VNM", "VPB", "VRE", "VIX", "SHB", "LPB"]
    selected_stocks = vn30_list
    st.sidebar.success("Đã chọn 30 cổ phiếu VN30")
else:
    st.sidebar.info("Đang lấy Top 100 mã thanh khoản cao nhất...")
    vn100_list = ["HPG","FPT","TCB","SSI","STB","VND","MWG","MBB","VHM","VIC","VPB","DIG","NVL","GEX","VCI","MSN","VNM","ACB","CTG","SHB","HDB","VIX","KBC","PDR","DXG","VCB","DGC","TPB","HSG","NKG","VRE","EIB","POW","GAS","LPB","TCH","VJC","BID","PLX","SAB","BVH","REE","PNJ","GVR","FRT","FTS","CTS","BSI","VHC","ANV","IDC","KDH","NLG","DBC","PVS","PVD","SCS","VOS","PVT","HAH","DCM","DPM","PC1","GEG","VGT","TNG","MSB","OCB","VIB","BAB","TTA","BCG","HDG","SAM","AAA","PHR","SZC","VPI","CII","HHV","LCG","VCG","LSS","SBT","QNS","MIG","GIL","VNA","SKG","VSC"]
    selected_stocks = vn100_list
    st.sidebar.success("Đã chọn 100 cổ phiếu thanh khoản cao nhất")

if st.sidebar.button("🚀 Chạy phân tích Multi-View", type="primary"):
    with st.spinner("Đang phân tích..."):
        results = []
        day_factor, day_note, near_exp, days_to_exp, exp_text = get_market_context()

        for symbol in selected_stocks:
            try:
                # Cách lấy dữ liệu an toàn nhất
                df = Vnstock().stock(symbol=symbol).quote.history(
                    start=(datetime.now() - timedelta(days=130)).strftime("%Y-%m-%d"),
                    end=datetime.now().strftime("%Y-%m-%d"),
                    interval="1D"
                )
                time.sleep(1.5)

                if df is None or df.empty or len(df) < 40:
                    continue

                latest = df.iloc[-1]
                current_price = latest['close']
                support = df['low'].rolling(20).min().iloc[-1]

                view_scores = calculate_view_scores(df, current_price, support)
                tech_score = calculate_weighted_score(view_scores)

                final_score = tech_score + day_factor
                if near_exp:
                    final_score += 1.0 if days_to_exp >= 0 else 0.5

                final_score = round(min(max(final_score, 3.0), 10.5), 2)

                fib_382, fib_50, fib_618 = calculate_fibonacci(df)

                results.append({
                    'Mã CK': symbol,
                    'Giá hiện tại': round(current_price, 2),
                    'Fib 38.2': fib_382,
                    'Fib 50': fib_50,
                    'Fib 61.8': fib_618,
                    'Trend': round(view_scores.get('Trend',0),1),
                    'Momentum': round(view_scores.get('Momentum',0),1),
                    'Oscillator': round(view_scores.get('Oscillator',0),1),
                    'Volume': round(view_scores.get('Volume',0),1),
                    'Volatility': round(view_scores.get('Volatility',0),1),
                    'PriceAction': round(view_scores.get('PriceAction',0),1),
                    'Ichimoku': round(view_scores.get('Ichimoku',0),1),
                    'Tech Score': tech_score,
                    'Final Score': final_score,
                    'Ngành nghề': get_sector(symbol),
                    'Khuyến nghị': 'MUA MẠNH' if final_score >= 8.5 else 'MUA' if final_score >= 7.2 else 'THEO DÕI'
                })

            except Exception as e:
                st.error(f"Lỗi {symbol}: {str(e)[:100]}")

        if results:
            df_result = pd.DataFrame(results)
            df_result = df_result.sort_values(by='Final Score', ascending=False).reset_index(drop=True)

            st.success(f"✅ Hoàn thành phân tích {len(results)} cổ phiếu!")

            st.subheader("🏆 Bảng Xếp Hạng Multi-View")
            st.dataframe(
                df_result.style.background_gradient(subset=['Final Score'], cmap='RdYlGn'),
                use_container_width=True,
                height=700,
                column_config={"Ngành nghề": st.column_config.TextColumn("Ngành nghề")}
            )

            # Tải Excel
            filename = f"MultiView_Trade_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
            df_result.to_excel(filename, index=False)
            with open(filename, "rb") as f:
                st.download_button("📥 Tải file Excel", data=f, file_name=filename)

with st.expander("📋 Hướng dẫn"):
    st.write("• Đã đăng ký API Key vnstock")
    st.write("• Final Score = Tech Score + Day Factor + Expiration Bonus")

st.caption("Phiên bản tối ưu cho trade ngắn hạn 5-7 ngày trên thị trường Việt Nam")
