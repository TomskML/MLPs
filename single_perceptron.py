def predict(inputs, weights):
	threshold = 0.0
	weighted_sum = 0.0

	for input, weight in zip(inputs,weights):
		# print("input is : ", type(input) , "and weight is :", type(weight))
		weighted_sum +=input*weight

	return 1.0 if weighted_sum >= threshold else 0.0 

def accuracy(inputs, weights):
	preds = []
	correct = 0.0
	for i in range(len(inputs)):
		pred = predict(inputs[i][:-1], weights)
		preds.append(pred)
		if pred == inputs[i][-1] : correct +=1

	print("predictions are : " , preds)

	return correct/float(len(inputs)) 


def train(inputs, weights, epochs = 10, learn_rate = 1.0 , stop_early=True):
	for i in range(epochs):
		curr_accuracy = accuracy(inputs,weights)
		print("for epoch %d  accuracy is %d " % (i , curr_accuracy))
		print("weights are ", weights)

		if curr_accuracy == 1.0 and stop_early : return 

		for i in range(len(inputs)):
			prediction = predict(inputs[i][:-1],weights)

			error = inputs[i][-1] - prediction
			for j in range(len(weights)):
				weights[j] = weights[j] + learn_rate*error*inputs[i][j]


	return weights


def main():
			#   bias   x1	, x2,	output
	data = [	[1.00, 0.08, 0.72, 1.0],
				[1.00, 0.10, 1.00, 0.0],
				[1.00, 0.26, 0.58, 1.0],
				[1.00, 0.35, 0.95, 0.0],
				[1.00, 0.45, 0.15, 1.0],
				[1.00, 0.60, 0.30, 1.0],
				[1.00, 0.70, 0.65, 0.0],
				[1.00, 0.92, 0.45, 0.0]]

	weights = [0.0 , 0.0 , 0.0 ]
	y= []
	y = train(data,weights, epochs=10, learn_rate=1.0, stop_early=True)

main()