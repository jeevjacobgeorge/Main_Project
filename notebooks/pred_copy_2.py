#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional


# In[3]:


# Import necessary libraries
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as spark_sum, col, hour, concat_ws, to_date, date_format
# Stop any existing Spark session
# Step 1: Initialize a Spark session
spark = SparkSession.builder \
    .appName("BigDataProcessing") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
# Step 2: Load your CSV file into a Spark DataFrame
data = spark.read.csv("June2024.csv", header=True, inferSchema=True)


# In[4]:


# Step 3: Convert 'TICKET_ISSUE_DATE' to the correct format (DD/MM/YYYY to YYYY-MM-DD)
data = data.withColumn("TICKET_ISSUE_DATE", to_date(col("TICKET_ISSUE_DATE"), "dd/MM/yyyy"))

# Step 4: Extract hour from 'TICKET_ISSUE_TIME' and combine with 'TICKET_ISSUE_DATE'
# Extract hour as a two-digit string and concatenate with the formatted date
data = data.withColumn("TICKET_ISSUE_HOUR", date_format(col("TICKET_ISSUE_TIME"), "HH"))
data = data.withColumn("DATE_HOUR", concat_ws(" ", date_format(col("TICKET_ISSUE_DATE"), "yyyy-MM-dd"), col("TICKET_ISSUE_HOUR")))

# Step 5: Group by 'DATE_HOUR' and calculate the sum for 'NO_OF_ADULT' and 'NO_OF_CHILD'
summary_data = data.groupBy("DATE_HOUR") \
    .agg(
        spark_sum("NO_OF_ADULT").alias("NO_OF_ADULT"),
        spark_sum("NO_OF_CHILD").alias("NO_OF_CHILD")
    )
# Step 5: Convert Spark DataFrame to pandas DataFrame for further processing
df = summary_data.toPandas()
df


# In[ ]:





# In[5]:


import pandas as pd
# Step 7: Convert 'DATE_HOUR' to datetime format, then set as the index
df['DATE_HOUR'] = pd.to_datetime(df['DATE_HOUR'], format='%Y-%m-%d %H')
df.set_index('DATE_HOUR', inplace=True)
df.sort_index(inplace=True)
print(df)


# In[6]:


# Step 7: Scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)
print(scaled_data)


# In[7]:


# Step 8: Stop the Spark session
spark.stop()


# In[8]:


import matplotlib.dates as mdates

# Adult Tickets
plt.figure(figsize=(10, 4))
adult_scaled = [x for x, y in scaled_data]
plt.plot(df.index, adult_scaled, label='No of Adult Tickets')
plt.title('Adult Tickets')
plt.xlabel('Date')
plt.ylabel('No of Adult Tickets')
plt.legend()

# Set date format to only day and month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
# Set major ticks to appear every 4 days
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=4))

plt.show()

# Child Tickets
plt.figure(figsize=(10, 4))
child_scaled = [y for x, y in scaled_data]
plt.plot(df.index, child_scaled, label='No of Child Tickets')
plt.title('Child Tickets')
plt.xlabel('Date')
plt.ylabel('No of Child Tickets')
plt.legend()


plt.show()


# In[9]:


# Step 9: Prepare Train and Test Data
split_ratio = 0.7
split_index = int(len(scaled_data) * split_ratio)     
# Create Train and Test datasets
Train = scaled_data[:split_index]
Test = scaled_data[split_index:]


# In[10]:


# Step 10: Create TimeseriesGenerator for training data
n_input = 24  # Use the last 24 hours for prediction
n_features = 2  # Number of features (NO_OF_ADULT and NO_OF_CHILD)

generator = TimeseriesGenerator(Train, Train, length=n_input, batch_size=1)


# In[11]:


# Define the LSTM model
model = Sequential([
    Bidirectional(LSTM(units=100, activation='tanh', return_sequences=True), input_shape=(n_input, n_features)),
    Bidirectional(LSTM(units=50, activation='tanh')),
    Dense(units=n_features, activation='linear')  # 'linear' is appropriate for regression
])


# In[12]:


# compile the model

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])


# In[13]:


# model training
# Train the model
epochs = 25  # More epochs can lead to better performance
history = model.fit(generator, epochs=epochs, verbose=1)


# In[14]:


# Step 14: Generate sequences for the test data
test_generator = TimeseriesGenerator(Test, Test, length=n_input, batch_size=1)

# Step 15: Make predictions
predictions = model.predict(test_generator)

# Step 16: Inverse transform the predictions and actual test data
predicted_values = scaler.inverse_transform(predictions)
actual_values = scaler.inverse_transform(Test[n_input:])  # Use Test[n_input:] for matching lengths


# In[15]:


# Step 17: Create a time index for the predicted values
time_index = pd.date_range(start=df.index[split_index + n_input], periods=len(predicted_values), freq='H')

# Step 18: Create a DataFrame for the predicted and actual values for better visualization
results = pd.DataFrame({
    'Predicted_NO_OF_ADULT': predicted_values[:, 0],
    'Predicted_NO_OF_CHILD': predicted_values[:, 1],
    'Actual_NO_OF_ADULT': actual_values[:, 0],
    'Actual_NO_OF_CHILD': actual_values[:, 1],
}, index=time_index)

# Step 19: Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(results.index, results['Predicted_NO_OF_ADULT'], label='Predicted NO_OF_ADULT', color='blue')
plt.plot(results.index, results['Actual_NO_OF_ADULT'], label='Actual NO_OF_ADULT', color='red', alpha=0.5)
plt.title('Passenger Demand Prediction vs Actual')
plt.xlabel('Date and Hour')
plt.ylabel('Number of Adult Passengers')
plt.legend()
plt.show()


