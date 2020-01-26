def predict(X):
    return w0 + w1 * X

if __name__== "__main__":
    with open('weights.txt', 'r') as f:
        w0, w1 = [float(x) for x in next(f).split()]
    
    print ("Enter the kilometrage:")
    X = float(input())

    y = predict(X)
    print ("The estimated car price is: %f" %y)
