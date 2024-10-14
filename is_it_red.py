import numpy as np


#rule
#1. r have to be more than 40 no matter what
#2. if g or b are 2.55 times less than r then it's red
#3. if g and b are 0 then it's red
def is_red(colour) -> int:
    r, g, b = colour
    if g == b == 0:
        return int(r >= 0.156)
    
    if g > 0 and b > 0:
        return int(r / g >= 2.55 and r / b >= 2.55)
    
    if g == 0:
        return int(r >= 2.55 * b)
    
    if b == 0:
        return int(r >= 2.55 * g)

    return 0

def normalized_colour(colour) -> list:
    return (round(colour[0]/255, 3), round(colour[1]/255, 3), round(colour[2]/255, 3))

#calculate the gini index using the famous formula
def gini_index(y:list) -> float:
    n = float(len(y))
    classes = set(y)

    score = 0.0
    for i in classes:
        n_i = y.count(i) #get number of the class i
        p_i = n_i/n #probablility of class i
        score += p_i ** 2 #^2 then sum

    return (1.0 - score) #gini

def test_split(index, thr, X_train, y_train):
    left_x, left_y, right_x, right_y = list(), list(), list(), list()
    for x,y in zip(X_train, y_train):
        if x[index] <= thr:
            left_x.append(x)
            left_y.append(y)
        else:
            right_x.append(x)
            right_y.append(y)
    return left_x, left_y, right_x, right_y

def get_split(x_train, y_train):
    
    b_index, b_value, b_score, b_groups = None, None, -1, None
    
    gini_s = gini_index(y_train)
    n = float(len(y_train))
    
    for index in range(len(x_train[0])):
        vals = sorted([row[index] for row in x_train])
        thrs = [vals[0]] + [(n1+n2)/2 for n1,n2 in zip(vals[:-1], vals[1:])] + [vals[-1]]

        for thr in thrs:
            groups = test_split(index, thr, x_train, y_train)
            left_x, left_y, right_x, right_y = groups
            
            gini_left = gini_index(left_y) #gini of the left side
            n_left = len(left_y)
            gini_right = gini_index(right_y) #gini of the right side
            n_right = len(right_y)
            
            gini_avg = ((n_left/n)*gini_left) + ((n_right/n)*gini_right)
            
            if (gini_s - gini_avg) > b_score:
                b_index, b_value, b_score, b_groups = index, thr, gini_s - gini_avg, groups
    return {'index':b_index, 'value':b_value, 'score':gini_s, 'groups':b_groups}

def to_terminal(outcomes):

    return max(outcomes, key=lambda x : outcomes.count(x))

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left_x, left_y, right_x, right_y = node['groups']
    del(node['groups'])
    # check for a no split
    if not left_y or not right_y:
        node['left'] = node['right'] = to_terminal(left_y + right_y)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left_y), to_terminal(right_y)
        return
    # process left child
    if len(left_y) <= min_size:
        node['left'] = to_terminal(left_y)
    else:
        node['left'] = get_split(left_x, left_y)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right_y) <= min_size:
        node['right'] = to_terminal(right_y)
    else:
        node['right'] = get_split(right_x, right_y)
        split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(x_train, y_train, max_depth, min_size):
    root = get_split(x_train, y_train)
    split(root, max_depth, min_size, 1)
    return root

# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d <= %.3f] gini = %.1f' % ((depth*' ', (node['index']), node['value'], node['score'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))

# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] <= node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def accuray_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        correct += actual[i] == predicted[i]
    return correct / float(len(actual)) * 100.0

def confusion_matrix(actual, predicted):
    y_g = set(actual)
    y_p = set(predicted)
    mat = [[0]*len(y_p) for i in range(len(y_g))]
    for i, yg in enumerate(y_g):
        for j, yp in enumerate(y_p):
            for k in range(len(actual)):
                if actual[k] == yg and predicted[k] == yp:
                    mat[i][j] += 1
    return mat

def print_confusion_matrix(matrix):
    print(f'            Prediction')
    print(f'Reference ', end=' ')
    for num in range(len(matrix)):
        print(f'{num+1:4.0f} ', end=' ')
    print('')
    for i, row in enumerate(matrix):
        print(f'      {i+1:3d}', end=' ')
        for num in row:
            print(f'{num:5.0f}', end=' ')
        print('')

def precision_recall(actual, predicted, eps=1e-6):
    tp = 0 
    for y_g, y_p in zip(actual, predicted):
        tp += (y_g == 1) & (y_p == 1)
    fp = 0
    for y_g, y_p in zip(actual, predicted):
        fp += (y_g == 0) & (y_p == 1)
    fn = 0 
    for y_g, y_p in zip(actual, predicted):
        fn += (y_g == 1) & (y_p == 0)
    tn = 0
    for y_g, y_p in zip(actual, predicted):
        tn += (y_g == 0) & (y_p == 0)
    
    precision = tp / (tp + fp) 
    recall = tp / (tp + fn) 
    return precision, recall

def f1score(actual, predicted, eps=1e-6):
    p, r = precision_recall(actual, predicted)
    return (2*p*r)/(p+r+eps)

X_train = [
    [1, 0, 0],
    [0.6, 0.05, 0.1],
    [0.16, 0, 0],
    [0.078, 0, 0],
    [1, 0.45, 0.39],
    [0.078, 0.039, 0.039],
    [0.16, 0.039, 0.039],
    [0.208, 0.078, 0.078],
    [0, 0, 0],
    [1, 1, 1],
    [0, 1, 1],
    [0.05, 0, 0],
    [0.1, 0, 0],
    [0.25, 1, 0],
    [0.31, 0, 1],
    [0.41, 1, 1],
    [0.1, 0.9, 0.9],
    [0.47, 0, 0],
    [0.9, 0.1, 0.1],
    [0.63, 0.63, 0.63],
    [0.13, 0.13, 0.13],
    [0.1, 0.1, 0.1],
    [0.988, 0.137, 0.196],
    [0.8, 0.2, 0.2],
    [0.75, 0.25, 0.2],
    [0.7, 0.1, 0.25]
]
y_train = [is_red(X_train[i]) for i in range(len(X_train))]

X_test = [
    [0.3, 0.1, 0.1],
    [0.4, 0, 0],
    [0.9, 0.3, 0.3],
    [0.1, 0, 0]
]
y_test = [is_red(X_test[i]) for i in range(len(X_test))]
print("\n"*4)
print(f"the training data contain r, g, b")
print(f"there are {len(X_train)} train data")
print(f"there are {len(X_test)} test data")

max_depth = 6
min_size = 10

tree = build_tree(X_train, y_train, max_depth, min_size)

if __name__ == "__main__":
    print("==========result after build tree==========")
    print("================train data================")
    y_predict = []
    for x, y in zip(X_train, y_train):
        prediction = predict(tree, x)
        y_predict.append(predict(tree, x))
    print("expected :", y_train)
    print("predict  :", y_predict)
    print("accuracy :", accuray_metric(y_train, y_predict))
    print("=================test data=================")
    y_predict = []
    for x in X_test:
        prediction = predict(tree, x)
        y_predict.append(predict(tree, x))

    print("expected :", y_test)
    print("predict  :", y_predict)
    print("accuracy :", accuray_metric(y_test, y_predict))
    mat = confusion_matrix(y_test, y_predict)
    print_confusion_matrix(mat)
    p, r = precision_recall(y_test, y_predict)
    print(f'precision = {p:.3f}')
    print(f'recall = {r:.3f}')
    f1 = f1score(y_test, y_predict)
    print(f'F1-score = {f1:.3f}')
    print("===========================================")