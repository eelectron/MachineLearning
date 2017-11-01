function y = runLR()
  minCost = inf
  x=[472 430 460 401 732 209 291 307 733 441 447 530 692 391 307 420 497]';
  x = [ones(size(x,1),1) x];
  y=[0 0 1 0 1 0 0  0 1 1 0 1 1 0 0 0 1]';
  toleranceVec = [1; 1];
  theta = [0; 0]; %Initial theta
  alpha = 0.5; %Small learning rate
  i=0;
  while i<1000
    [curCost, grad] = costFunction(theta, x, y);
    newTheta = theta - alpha*grad; %new theta approximated  
    diff = abs(theta - newTheta) < .0001; 
    if  isequal(diff, toleranceVec)
      break
    end
    theta = newTheta; 
    if curCost < minCost
      minCost = curCost;
    end
    i++;
  end
  
  minCost
  theta
  %Now predict application for 450
  v = [1 450]*theta;
  sigmoid(v)
end