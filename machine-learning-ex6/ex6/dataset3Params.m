function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
CAndSigma = [.01 .03 0.1 0.3 1 3 10 30];
cvErr = inf;    %%initial error is infinity as we train it will reduce.


for i = 1:length(CAndSigma)
    for j = 1:length(CAndSigma)
        %train with current C and sigma
        model = svmTrain(X, y, CAndSigma(i), @(x1, x2)gaussianKernel(x1, x2, CAndSigma(j)));
        
        %compute prediction
        predictions = svmPredict(model, Xval);
        
        %compute error
        err = mean(double(predictions ~= yval)) ;       
        
        if err < cvErr      %if err with current c, sg is less than prev then save current c, sg
            C = CAndSigma(i);
            sigma = CAndSigma(j);
            cvErr = err;
        end
    end
end



% =========================================================================

end
