import timesynth as ts
import streamlit as st
from bokeh.plotting import figure
import pandas as pd
import numpy as np
    

def generate_data(process_type = "Harmonic", stop_time = 1, num_points = 50, keep_percentage = 50, irregular = True, std_noise =0.3, **kwargs):
    time_sampler = ts.TimeSampler(stop_time = stop_time)
    if irregular:
        time_samples = time_sampler.sample_irregular_time(num_points = num_points, keep_percentage = keep_percentage)
    else:
        time_samples = time_sampler.sample_regular_time(num_points = num_points)
    
    if process_type == "Harmonic":
        signal = ts.signals.Sinusoidal(frequency = kwargs.get("frequency")) # number of sinusoids
    if process_type == "GaussianProcess":
        if kwargs.get("kernel") == "Constant":
            signal = ts.signals.GaussianProcess(kernel = kwargs.get("kernel"), variance = kwargs.get("variance"))
        if kwargs.get("kernel") == "SE":
            signal = ts.signals.GaussianProcess(kernel = kwargs.get("kernel"))
    if process_type == "PseudoPeriodic":
        signal = ts.signals.PseudoPeriodic(frequency = kwargs.get("frequency"), freqSD = kwargs.get("freqSD"), ampSD = kwargs.get("ampSD"))
    
    if kwargs.get("noise_type") == "White":
        noise = ts.noise.GaussianNoise(std=std_noise)
    if kwargs.get("noise_type") == "Red":
        noise = ts.noise.RedNoise(std=std_noise)
    else:
        noise = ts.noise.GaussianNoise(std=std_noise)
    timeseries = ts.TimeSeries(signal, noise_generator = noise)
    samples, signals, errors = timeseries.sample(time_samples)
    return time_samples, samples

def convert_df(df):
    return df.to_csv().encode('utf-8')

def plot_TS(c, time_samples, samples):
    if len(time_samples)>0:
        p = figure(
                 title='TS',
                 x_axis_label='x',
                 y_axis_label='y',
                 max_height=300,
                 height_policy='max')

        p.line(time_samples, samples, legend_label='Regular', line_width=2)
        p.circle(time_samples, samples, legend_label='Regular', line_width=2, fill_color='blue', size=5)
        c.bokeh_chart(p, use_container_width=True)

description = "Synthetic TS Generation"

# Your app goes in the function run()
def run():
    st.subheader("Synthetic TS Generation")
    c = st.container()
    
    with st.sidebar:
        st.write("-------------------------")
        
        option = st.selectbox('Which process?',("Harmonic", "GaussianProcess", "PseudoPeriodic", "AutoRegressive" ,"CAR","NARMA"))
        num_points = st.slider('Number of points', 0, 1000, 100, 5)
        num_ts = st.slider('Number of TS', 1, 1000, 1, 5)
        MAX_TO_PLOT = st.slider('Max number of TS to Plot', 1, 10, 1, 1)
        keep_percentage = 50 
        irregular = st.radio('Irregular', ("True", "False"))
        if irregular == "True":
            irregular = True
        else: 
            irregular = False
        noise_type = st.radio("Noise", ('White', 'Red'))
        std_noise = st.slider('Std of the noise', 0.0, 1.0, 0.3, 0.01)
        time_samples, samples = [], []
        
        if option == "Harmonic":
            df = pd.DataFrame(columns=["ID","x","y"])
            df_temp = pd.DataFrame(columns=["ID","x","y"])
            st.write("Signal generator for harmonic (sinusoidal) waves")
            amplitude = st.slider("Amplitude - Amplitude of the harmonic series", 0.0, 10.0, 1.0, 0.1)
            frequency = st.slider('Frequency - Frequency of the harmonic series', 0.0, 100.0, 1.0, 0.1)
            for i in range(num_ts):
                time_samples, samples = generate_data(process_type = option,
                                                      num_points = num_points*2, 
                                                      irregular = irregular,
                                                      std_noise = std_noise,
                                                      frequency = frequency
                                                    )
                
                df_temp.x = time_samples
                df_temp.y = samples
                df_temp.ID = np.full(len(samples), i)
                
                df = pd.concat([df,df_temp], axis = 0, ignore_index=True)
                
                if(i < MAX_TO_PLOT):
                    plot_TS(c, time_samples, samples)
                
    
        if option == "GaussianProcess":
            df = pd.DataFrame(columns=["ID","x","y"])
            df_temp = pd.DataFrame(columns=["ID","x","y"])
            kernel = st.radio("Kernel",('SE', 'Constant', 'Exponential', 'RQ', 'Linear', 'Matern', 'Periodic'))
            
            if kernel == "SE": # the squared exponential
                for i in range(num_ts):
                    time_samples, samples = generate_data(process_type = option,
                                                      num_points = num_points*2, 
                                                      irregular = irregular,
                                                      std_noise = std_noise,
                                                      kernel = kernel
                                                )
                    df_temp.x = time_samples
                    df_temp.y = samples
                    df_temp.ID = np.full(len(samples), i)

                    df = pd.concat([df,df_temp], axis = 0, ignore_index=True)
                    
                    if(i < MAX_TO_PLOT):
                        plot_TS(c, time_samples, samples)
            
            if kernel == "Constant": # All covariances set to `variance`
                variance = st.slider('Variance', 0.0, 1.0, 1.0, 0.1)
                for i in range(num_ts):
                    time_samples, samples = generate_data(process_type = option,
                                                      num_points = num_points*2, 
                                                      irregular = irregular,
                                                      std_noise = std_noise,
                                                      kernel = kernel,
                                                      variance = variance
                                                )
                    df_temp.x = time_samples
                    df_temp.y = samples
                    df_temp.ID = np.full(len(samples), i)

                    df = pd.concat([df,df_temp], axis = 0, ignore_index=True)
                    
                    if(i < MAX_TO_PLOT):
                        plot_TS(c, time_samples, samples)
                    
        if option == "PseudoPeriodic":
            df = pd.DataFrame(columns=["ID","x","y"])
            df_temp = pd.DataFrame(columns=["ID","x","y"])
            st.write("Signal generator for pseudoeriodic waves. (The wave's amplitude and frequency have some stochasticity that can be set manually.)")
            amplitude = st.slider('amplitude - Amplitude of the harmonic series', 0.0, 10.0, 1.0, 0.1)
            frequency = st.slider('Frequency - Frequency of the harmonic series', 0.0, 100.0, 1.0, 0.5)
            ampSD = st.slider('ampSD - Amplitude standard deviation', 0.0, 1.0, 0.1, 0.01)
            freqSD = st.slider('freqSD - Frequency standard deviation', 0.0, 1.0, 0.1, 0.01)
            
            for i in range(num_ts):
                time_samples, samples = generate_data(process_type = option,
                                                  num_points = num_points*2, 
                                                  irregular = irregular,
                                                  std_noise = std_noise,
                                                  frequency = frequency,
                                                  freqSD = freqSD,
                                                  ampSD = ampSD,
                                                  amplitude =  amplitude
                                                )
                
                df_temp.x = time_samples
                df_temp.y = samples
                df_temp.ID = np.full(len(samples), i)

                df = pd.concat([df,df_temp], axis = 0, ignore_index=True)
                
                if(i < MAX_TO_PLOT):
                    plot_TS(c, time_samples, samples)
        
        csv = convert_df(df)
        st.download_button(
               "Press to Download",
               csv,
               "data.csv",
               "text/csv",
               key='download-csv'
            )
    
            
    
    
        
        
    
# end of app

# This code allows you to run the app standalone
# as well as part of a library of apps
if __name__ == "__main__":
    run()