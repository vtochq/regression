with open('dash_12_10_18.json') as f:
    data_raw = json.load(f)

dataset2 = np.zeros((len(data_raw),7))
for i in range(0,len(data_raw)):
    dataset2[i]=[data_raw[i]['high'], data_raw[i]['low'], data_raw[i]['open'], data_raw[i]['close'],\
    data_raw[i]['volume']/10000, data_raw[i]['quoteVolume']/150000, data_raw[i]['weightedAverage']]

high_prices2 = dataset2[:,0]
low_prices2 = dataset2[:,1]
mid_prices2 = (high_prices2+low_prices2)/2.0

scaler2 = MinMaxScaler()
mid_prices2 = mid_prices2.reshape(-1,1)
#test_data = test_data.reshape(-1,1)

smoothing_window_size = 2500
for di in range(0,mid_prices2.size,smoothing_window_size):
    scaler2.fit(mid_prices2[di:di+smoothing_window_size,:])
    mid_prices2[di:di+smoothing_window_size,:] = scaler2.transform(mid_prices2[di:di+smoothing_window_size,:])

scaler2.fit(mid_prices2[di+smoothing_window_size:,:])
mid_prices2[di+smoothing_window_size:,:] = scaler2.transform(mid_prices2[di+smoothing_window_size:,:])

mid_prices2 = mid_prices2.reshape(-1)

plt.plot(range(len(mid_prices2)), mid_prices2, 'y')
plt.plot(range(len(mid_prices)), mid_prices, 'r')
plt.show()

for w_i in test_points_seq:
  mse_test_loss = 0.0
  our_predictions = []
  x_axis=[]
  # Feed in the recent past behavior of stock prices
  # to make predictions from that point onwards
  for tr_i in range(w_i-num_unrollings+1,w_i-1):
    current_price = all_mid_data[tr_i]
    feed_dict[sample_inputs] = np.array(current_price).reshape(1,1)
    _ = session.run(sample_prediction,feed_dict=feed_dict)
  feed_dict = {}
  current_price = all_mid_data[w_i-1]
  feed_dict[sample_inputs] = np.array(current_price).reshape(1,1)
  # Make predictions for this many steps
  # Each prediction uses previous prediciton as it's current input
  for pred_i in range(n_predict_once):
    pred = session.run(sample_prediction,feed_dict=feed_dict)
    our_predictions.append(np.asscalar(pred))
    feed_dict[sample_inputs] = np.asarray(pred).reshape(-1,1)
    x_axis.append(w_i+pred_i)
    try:
        mse_test_loss += 0.5*(pred-all_mid_data[w_i+pred_i])**2
    except:
        pass
  session.run(reset_sample_states)
  predictions_seq.append(np.array(our_predictions))
  mse_test_loss /= n_predict_once
  mse_test_loss_seq.append(mse_test_loss)
  x_axis_seq.append(x_axis)


current_test_mse = np.mean(mse_test_loss_seq)
# Learning rate decay logic
if len(test_mse_ot)>0 and current_test_mse > min(test_mse_ot):
    loss_nondecrease_count += 1
else:
    loss_nondecrease_count = 0

if loss_nondecrease_count > loss_nondecrease_threshold:
      session.run(inc_gstep)
      loss_nondecrease_count = 0
      print('\tDecreasing learning rate by 0.5')

test_mse_ot.append(current_test_mse)
print('\tTest MSE: %.5f'%np.mean(mse_test_loss_seq))
predictions_over_time.append(predictions_seq)
print('\tFinished Predictions')


best_prediction_epoch = len(predictions_over_time)-1 # replace this with the epoch that you got the best results when running the plotting code

plt.figure(figsize = (18,18))
plt.subplot(2,1,1)
plt.plot(range(dataset.shape[0]),all_mid_data,color='b')

# Plotting how the predictions change over time
# Plot older predictions with low alpha and newer predictions with high alpha
start_alpha = 0.25
alpha  = np.arange(start_alpha,1.1,(1.0-start_alpha)/len(predictions_over_time[::3]))
for p_i,p in enumerate(predictions_over_time[::3]):
    for xval,yval in zip(x_axis_seq,p):
        plt.plot(xval,yval,color='r',alpha=alpha[p_i])

plt.title('Evolution of Test Predictions Over Time',fontsize=14)
plt.xlabel('Date')
plt.ylabel('Mid Price')
#plt.xlim(11000,12500)

plt.subplot(2,1,2)

# Predicting the best test prediction you got
plt.plot(range(dataset.shape[0]),all_mid_data,color='b')
for xval,yval in zip(x_axis_seq,predictions_over_time[best_prediction_epoch]):
    plt.plot(xval,yval,color='r')

plt.title('Best Test Predictions Over Time',fontsize=14)
plt.xlabel('Date')
plt.ylabel('Mid Price')
#plt.xlim(11000,12500)
plt.show()
