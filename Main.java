import Jama.Matrix;

/**
 * A basic Machine Learning application to predict how much rent a property in
 * Dublin should be, given a few parameters. It uses JAMA, a Java matrix package
 * to do some of the heavy lifting.
 * 
 * @author Michael McMullin
 *
 */
public class Main {
    public static void main(String[] args) {
        // The training data (i.e. data where we already know what results to
        // expect). In this case, it's various features of a rental property in
        // Dublin. Each feature needs to be a number that can be compared with
        // other training rows. This can be stored in a 2-dimensional array.
        // In this case, each row contains a number for:
        // * Number of bedrooms
        // * Number of bathrooms
        // * Is the property in north Dublin? (10 = yes, 0 = no)
        // * Is the property an apartment? (10 = yes, 0 = no)
        // * Is the property a house? (10 = yes, 0 = no)
        // * Size of the property (sq m)
        // * Size of the property squared <- 'feature engineering' to improve accuracy.
        // * Distance from O'Connell Bridge (in km)
        double[][] trainingData = {
                { 2, 2,  0, 10,  0,  91.00,  8281.00, 13.05 }, // http://www.daft.ie/dublin/apartments-for-rent/killiney/apartment-28-killiney-hill-killiney-dublin-1791291/
                { 2, 3, 10, 10,  0, 122.00, 14884.00,  2.28 }, // http://www.daft.ie/dublin/apartments-for-rent/drumcondra/the-garden-house-waterfall-avenue-richmond-rd-drumcondra-dublin-1797648/
                { 2, 2,  0, 10,  0, 102.19, 10442.80,  8.79 }, // http://www.daft.ie/dublin/apartments-for-rent/monkstown/the-maples-monkstown-valley-monkstown-dublin-1796263/
                { 3, 2,  0, 10,  0, 127.00, 16129.00,  3.33 }, // http://www.daft.ie/dublin/apartments-for-rent/ballsbridge/lynton-court-merrion-road-ballsbridge-dublin-1796372/
                { 2, 2,  0, 10,  0,  70.00,  4900.00,  1.10 }, // http://www.daft.ie/dublin/apartments-for-rent/dublin-2/garden-apartment-lower-baggot-street-dublin-2-dublin-1796365/
                { 1, 1, 10, 10,  0,  44.00,  1936.00,  1.03 }, // http://www.daft.ie/dublin/apartments-for-rent/ifsc/semple-house-custom-house-square-mayor-street-ifsc-dublin-1795645/
                { 2, 1, 10, 10,  0,  52.00,  2704.00,  1.21 }, // http://www.daft.ie/dublin/apartments-for-rent/ifsc/slaney-house-custom-house-square-ifsc-dublin-1791839/
                { 4, 2,  0, 10,  0, 127.00, 16129.00,  3.28 }, // http://www.daft.ie/dublin/apartments-for-rent/ballsbridge/8-lynton-court-ballsbridge-dublin-1797252/
                { 1, 1, 10, 10,  0,  48.00,  2304.00,  2.96 }, // http://www.daft.ie/dublin/apartments-for-rent/ifsc/aran-house-custom-house-square-ifsc-dublin-1797937/
                { 2, 2,  0, 10,  0,  80.00,  6400.00, 13.41 }, // http://www.daft.ie/dublin/apartments-for-rent/cabinteely/garrison-mews-cabinteely-dublin-1797721/
                { 2, 2,  0, 10,  0,  75.00,  5625.00,  1.88 }, // http://www.daft.ie/dublin/apartments-for-rent/dublin-2/grand-canal-square-dublin-2-dublin-1732078/
                { 2, 2,  0, 10,  0,  75.00,  5625.00,  1.65 }, // http://www.daft.ie/dublin/apartments-for-rent/dublin-2/iveagh-court-dublin-2-dublin-1732071/
                { 1, 1,  0, 10,  0,  50.00,  2500.00,  1.44 }, // http://www.daft.ie/dublin/apartments-for-rent/dublin-2/grand-canal-dock-dublin-2-dublin-1732799/
                { 4, 3, 10,  0, 10, 139.00, 19321.00,  5.33 }, // http://www.daft.ie/dublin/houses-for-rent/dublin-7/navan-road-north-dublin-city-dublin-7-dublin-1794392/
                { 2, 1, 10,  0, 10,  73.00,  5329.00,  1.87 }, // http://www.daft.ie/dublin/houses-for-rent/east-wall/caledon-court-east-road-east-wall-dublin-1797320/
                { 2, 2,  0, 10,  0,  98.00,  9604.00,  2.51 }, // http://www.daft.ie/dublin/apartments-for-rent/ballsbridge/shelbourne-hall-shelbourne-road-ballsbridge-dublin-1794204/
                { 2, 2,  0, 10,  0,  72.00,  5184.00,  5.68 }, // http://www.daft.ie/dublin/apartments-for-rent/goatstown/trimbleston-goatstown-dublin-14-goatstown-dublin-1791411/
                { 2, 1,  0,  0, 10,  63.00,  3969.00,  1.92 }, // http://www.daft.ie/dublin/houses-for-rent/ranelagh/dartmouth-place-ranelagh-dublin-1793402/
        };
        
        // Training data needs to have a set of results, one per property. These
        // can be stored in a single-dimension array. For each property above,
        // the results are the monthly rents. Once the system has been 'trained',
        // we can take a new set of features, and see if it can predict an
        // accurate rent.
        double[] trainingResults = {
                2250,
                2000,
                1924,
                3300,
                2950,
                1660,
                2100,
                3500,
                1600,
                1600,
                3950,
                4250,
                3450,
                2995,
                2000,
                3700,
                1900,
                2500
        };
        
        // This array uses the 'train' method, which calculates a series of
        // weights that can be applied to new data to predict new results (in
        // this case, what the monthly rent should be).
        double[][] theta = train(trainingData, trainingResults);

        // ---------------------------------------------------------------------
        // Now that an approximate pattern has been calculated between features
        // and rents, try predicting a few new property rental prices. In fact,
        // we know what the results should be in these cases, but we're not telling
        // the prediction system what they are. We're just using them to compare
        // against the prediction results so we know how accurate the system is.
        double[][] testData = {
                { 2, 2,  0, 10, 0, 157.93, 24941.8849, 2.33 }, // Actual price: 3500 -- http://www.daft.ie/dublin/houses-for-rent/ballsbridge/wellington-place-ballsbridge-dublin-1796683/
                { 2, 1,  0, 10, 0,  97.00,  9409.0000, 1.44 }, // Actual price: 3000 -- http://www.daft.ie/dublin/apartments-for-rent/dublin-2/4a-lad-lane-lower-baggot-street-dublin-2-dublin-1792896/
                { 1, 1, 10, 10, 0,  48.00,  2304.0000, 1.00 }  // Actual price: 1850 -- http://www.daft.ie/dublin/apartments-for-rent/ifsc/block-3-clarion-quay-ifsc-dublin-1797340/
        };
        double[] testResults = { 3500, 3000, 1850 };

        // Loop through the new test data, and predict what the rent should be.
        // Then compare against the actual rent to see how accurate it is.
        double avg = 0;
        for (int i=0; i<testData.length; i++) {
            double price1 = predict(testData[i], theta);
            double price2 = testResults[i];
            double diff = price1 - price2;
            avg += diff;
            System.out.printf("Prediction for property %d rent is: %.0f. Actual rent: %.0f  (error: %.0f Euro)\n",
                    i+1, price1, price2, diff);
        }
        System.out.printf("\nAverage error: %.0f Euro", avg/testResults.length);
    }
    
    // =========================================================================
    // END OF MAIN METHOD
    // Everything below here is supporting methods. If you've little interest in
    // linear algebra, feel free to ignore!
    // =========================================================================

    
    /**
     * Utility method to print out a 2d array to the console.
     * @param arr   The array to print out.
     */
    static void printArray(double[][] arr) {
        for (int i=0; i<arr.length; i++) {
            for (int j=0; j<arr[0].length; j++) {
                System.out.printf("%.2f\t", arr[i][j]);
            }
            System.out.println();
        }
    }

    /**
     * Calculate the relationship be
     * @param trainingData    2d array containing features for each sample.
     * @param trainingResults Array containing the results for each sample.
     * @return An array of weights describing a multi-dimensional line that
     * approximates the supplied data.
     */
    static double[][] train(double[][] trainingData, double[] trainingResults) {
        Matrix A = new Matrix(trainingData);
        Matrix B = new Matrix(trainingResults, trainingResults.length);

        return getNormalEquationResult(A, B).getArray();
    }

    /**
     * According to Wolfram MathWorld, "the normal equation is that which
     * minimizes the sum of the square differences between the left and right
     * sides". In other words, the equation of a line that approximately follows
     * the supplied data.
     * @param X  A Matrix containing the training data.
     * @param Y  A Matrix (vector) containing the training results.
     * @return A Matrix 'theta' that describe the required line.
     */
    static Matrix getNormalEquationResult(Matrix X, Matrix Y) {
        // Original MATLAB / Octave formula from Andrew Ng's introductory ML course:
        // theta = inv(X' * X) * X' * y
        Matrix output;

        Matrix transposeX = X.transpose();
        Matrix x2 = transposeX.times(X);
        Matrix x3 = x2.inverse();

        output = x3;
        output = output.times(transposeX.times(Y));

        return output;
    }

    /**
     * For a new set of features, predict what the corresponding result should
     * be, based on the results of the training process.
     * @param X  An array of features for a new sample (in this case, a new
     *           property that didn't appear in the training data).
     * @param theta  The weights to apply to each feature to calculate a result.
     * @return   The predicted result of the new data. In this case, the
     *           predicted rent for the supplied property features.
     */
    static double predict(double[] X, double[][] theta) {
        Matrix xMatrix = new Matrix(X, 1);
        Matrix thetaMatrix = new Matrix(theta);
        return xMatrix.times(thetaMatrix).getArray()[0][0];
    }
}
