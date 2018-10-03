# Dublin Rent Predictor

This is a simple Java console application that grew out of a lecture we had
in C2 about arrays. It makes use of the [JAMA Java matrix package](https://math.nist.gov/javanumerics/jama/)
to do some of the heavy lifting implementing the normal equation for linear
regression.

The parameters are supplied as a 2-D array (`trainingData`), with features as
indicated in the comments below. These are rental properties where we already
know the prices: a second single-dimension array stores the prices themselves
for each of these samples. By using a technique called 'linear regression',
we can 'train' the application to model this data.

With the results of the training, you can call the `predict` method with an
array of features for a new property, and it will try to predict how much the
rent will be. For the three examples included here, the output was:

**Prediction for property 1 rent is: 3442. Actual rent: 3500  (error: -58 Euro)  
Prediction for property 2 rent is: 3073. Actual rent: 3000  (error: 73 Euro)  
Prediction for property 3 rent is: 1843. Actual rent: 1850  (error: -7 Euro)**

As with most machine learning algorithms, the results are rarely 100% accurate,
but the samples above are still reasonably good. Other samples may vary.
Choosing the right features to use will improve accuracy, but is a tough task
in itself. Lots more training data would be better too (this data was pulled
from daft.ie at random on Wed 13th December 2017, only including properties
that gave the property size, which I felt was an important feature to include).

Feel free to play around with the arrays to solve other prediction problems.
The array `trainingData` should define whatever features you're measuring, with
each row representing a different sample. `trainingResults` can be any result
set that can be represented as a single number. Just make sure the length
matches the number of rows in `trainingData`. Finally, the `testData` array
represents a new sample whose result is unknown and needs to be predicted.
Make sure it has the same number of features as each sample in the
`trainingData` array. Check the comments throughout the code for more info.
