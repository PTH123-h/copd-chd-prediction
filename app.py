import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. 网页标题和基本设置
st.set_page_config(page_title="老年COPD并发CHD风险预测", page_icon="🫁")
st.title("🫁 老年COPD并发CHD风险预测模型")
st.markdown("""
**简介：** 本工具用于预测老年慢性阻塞性肺疾病（COPD）患者并发冠心病（CHD）的风险。
**注意：** 本工具仅供医学研究与交流使用，不作为临床最终诊断依据。
""")

# 2. 加载随机森林模型
@st.cache_resource
def load_model():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"模型加载失败，请确认 rf_model.pkl 文件在同级目录下。错误信息: {e}")
    st.stop()

# 3. 侧边栏：输入患者特征
st.sidebar.header("请填入患者的临床指标")

def user_input_features():
    # 分类变量
    # 注意：请根据你训练模型时的设定调整，通常 0=女，1=男，如果相反请对调这里的逻辑
    gender_option = st.sidebar.selectbox('性别 (GENDER)', ['男 (Male)', '女 (Female)'])
    gender = 1 if gender_option == '男 (Male)' else 0

    # 连续变量 (这里的默认值、最大最小值你可以根据实际医学单位和范围进行微调)
    age = st.sidebar.number_input('年龄 (AGE, 岁)', min_value=40, max_value=120, value=75)
    
    # 实验室指标 (假设常用单位，请务必核对是否与你训练数据单位一致)
    ua = st.sidebar.number_input('尿酸 (UA, μmol/L)', min_value=0.0, max_value=1000.0, value=350.0)
    hs_crp = st.sidebar.number_input('超敏C反应蛋白 (hs_CRP, mg/L)', min_value=0.0, max_value=200.0, value=5.0)
    glu = st.sidebar.number_input('空腹血糖 (Glu, mmol/L)', min_value=0.0, max_value=50.0, value=5.5)
    bun = st.sidebar.number_input('血尿素氮 (BUN, mmol/L)', min_value=0.0, max_value=100.0, value=6.0)
    hs_ctni = st.sidebar.number_input('超敏肌钙蛋白I (hs_cTnI, pg/mL)', min_value=0.0, max_value=10000.0, value=15.0)
    tt = st.sidebar.number_input('凝血酶时间 (TT, 秒)', min_value=0.0, max_value=100.0, value=18.0)

    # ！！！关键：这里的字典 Key 必须和你的变量名严格一致（大小写必须完全相同）！！！
    # 你的变量：'GENDER', 'UA', 'hs_CRP', 'Glu', 'AGE', 'BUN', 'hs_cTnI', 'TT'
    data = {
        'GENDER': gender,
        'UA': ua,
        'hs_CRP': hs_crp,
        'Glu': glu,
        'AGE': age,
        'BUN': bun,
        'hs_cTnI': hs_ctni,
        'TT': tt
    }
    features = pd.DataFrame(data, index=[0])
    
    # 确保列的顺序与你提供的完全一致
    cols_order = ['GENDER', 'UA', 'hs_CRP', 'Glu', 'AGE', 'BUN', 'hs_cTnI', 'TT']
    features = features[cols_order]
    
    return features

input_df = user_input_features()

# 4. 在主界面展示用户的输入
st.subheader('患者当前输入的指标：')
st.write(input_df)

# 5. 预测按钮及结果展示
st.divider()
if st.button('点击预测并发冠心病(CHD)风险'):
    try:
        # 提取预测概率 (类别1的概率)
        prediction_prob = model.predict_proba(input_df)[:, 1][0]
        
        st.subheader('预测结果')
        # 放大显示概率
        st.metric(label="并发CHD的概率", value=f"{prediction_prob:.1%}")
        
        # 进度条
        st.progress(float(prediction_prob))
        
        # 风险判断 (你指定的阈值 0.5)
        cutoff = 0.50
        if prediction_prob >= cutoff:
            st.error(f'⚠️ **高风险 (High Risk)**：该患者并发冠心病(CHD)的预测概率大于或等于 {cutoff*100}%。')
        else:
            st.success(f'✅ **低风险 (Low Risk)**：该患者并发冠心病(CHD)的预测概率低于 {cutoff*100}%。')

    except Exception as e:
        st.error(f"预测出错啦: {e}")
        st.warning("请检查训练模型时使用的库版本或特征名是否完全对应。")