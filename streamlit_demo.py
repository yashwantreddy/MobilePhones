import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from altair import Color, Scale

df = pd.read_csv("Mobile Classification Data/train.csv")

st.markdown(
    "<h1 style='text-align: center; color: MediumSeaGreen;'>Mobile Phone Specs EDA</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center; color: #2596be; font-size: 19px;'><i>by Yashwant Jankay</i></h1>",
    unsafe_allow_html=True,
)


st.subheader("Here is a snapshot of the dataset:")
st.write(df.head())

st.markdown(
    """
| Column Name        | Description   |
| :-------------:    |:-------------:| 
| battery_power      | Total energy a battery can store in one time measured in mAh | 
| blue               | Has bluetooth or not      | 
| clock_speed        | speed at which microprocessor executes instructions      | 
| dual_sim        | Has dual sim support or not      | 
| fc        | Front Camera mega pixels      | 
| four_g        | Has 4G or not      | 
| int_memory        | Internal Memory in Gigabytes      | 
| m_dep        | Mobile Depth in cm      | 
| mobile_wt        | Weight of mobile phone      | 
| n_cores        | Number of cores of processor      | 
| pc        | Primary Camera mega pixels      | 
| px_height        | Pixel Resolution Height      | 
| px_width        | Pixel Resolution Width      | 
| ram        | Random Access Memory in Mega Bytes      | 
| sc_h        | Screen Height of mobile in cm      | 
| sc_w        | Screen Width of mobile in cm      | 
| talk_time        | longest time that a single battery charge will last on a phone call      | 
| three_g        | Has 3G or not      | 
| touch_screen        | Has touch screen or not      | 
| wifi        | Has wifi or not      | 
| price_range        | This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost)     | 

"""
)

st.markdown("\n")
st.markdown("\n")

fig1 = plt.figure()
sns.boxenplot(df.price_range, df.ram)
plt.ylabel("RAM (in MB)")
plt.xlabel("Price")
plt.xticks(ticks=[0, 1, 2, 3], labels=["Low", "Medium", "High", "Very High"])

st.markdown(
    "<span style='color:MediumVioletRed'><u> **Exploratory Data Analysis:** </u></span>",
    unsafe_allow_html=True,
)
st.markdown(
    "Our primary focus will be - determing the **_relationships among features_**."
)
st.markdown("_Conclusively, we will evaluate whether this dataset is real or not._")

st.markdown("Let's start with a Box plot of **Price** vs **RAM**")

st.write(fig1)

st.markdown(
    "RAM size and phone price are *__positively correlated__*! This makes intuitive sense."
)

st.markdown(
    "Now a look at a Bar plot of **Number of phones that have a Touch Screen** and **color coding them with their respective price ranges**:"
)

fig2 = plt.figure()
sns.set_style("whitegrid")
sns.countplot(df.touch_screen, hue=df.price_range, palette="spring")
plt.xlabel("Touch Screen")
plt.xticks(ticks=[0, 1], labels=["No", "Yes"])
plt.legend(
    labels=["Low", "Moderate", "High", "Very High"],
    shadow=True,
    loc="lower right",
    title="Price",
    fontsize="small",
)

st.write(fig2)

st.markdown(
    "It appears that phones with touchscreens and no touchscreens are almost equally \
    distributed among different price ranges. However, **it makes little sense that there\
    are such a large number of _high_ and _very high_ priced phones with no touchscreens!** \
    (in fact higher than their corresponding _low_ and _modernate_ price categories)"
)

st.markdown(
    "Next we are going to engineer a new feature! Dividing the pixel \
    resolution height (*px_height*) by screen height in cm (*sc_h*) - we get \
    pixels per cm of height.For simplicity, we will call this ** ppcm ** . "
)

df["ppcm"] = df.px_height / df.sc_h

st.write(df[["px_height", "sc_h", "ppcm"]].head())

st.markdown("Let's observe the median of ppcm of different price categories-")

fig3 = plt.figure()
sns.set_style("darkgrid")
sns.barplot(df.price_range, df.ppcm, palette="ocean")
plt.ylabel("Median pixels per cm")
plt.xlabel("Price")
plt.xticks(ticks=[0, 1, 2, 3], labels=["Low", "Medium", "High", "Very High"])

st.write(fig3)

st.markdown(
    "_Low_ priced phones have a lower pixels per cm as compared to _High_ and _Very \
    High_ priced phones. With the excpetion of _Moderately_ priced phones - where the \
    median of ppcm is higher than that of _High_ priced phones, there seems to be a clear \
    upward trend - **higher priced phones have a higher pixel density, which is the  \
    result of a sharper screen resolution.** This makes perfect intuitive sense as well!"
)


# figx = alt.Chart(df).mark_point().encode(alt.X("pc:Q"), alt.Y("fc:Q"),)

# st.altair_chart(figx, use_container_width=True)

fig4 = plt.figure()
sns.set_style("whitegrid")
sns.pointplot(
    df.four_g,
    df.clock_speed,
    hue=df.price_range,
    scale=1.3,
    dodge=True,
    palette="Set1",
)
plt.ylabel("Processor Clock Speed (GHz)")
plt.xlabel("4G Enabled")
plt.xticks(ticks=[0, 1], labels=["No", "Yes"])
plt.legend(
    shadow=True, title="Price", fontsize="small",
)

st.write(fig4)

st.markdown(
    "An interesting trend that can be observed here is that category **_3_** priced \
        (very high priced) phones have a higher clock speed on non - 4G enabled phones as \
        compared to 4G enabled phones. \
    This seems a little weird - as 4G enabled phones need faster (higher) clock rates\
        than non - 4G enabled phones. This trend appears to be inverse for very high priced\
            phones."
)

