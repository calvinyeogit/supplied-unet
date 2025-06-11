from keras import backend as K

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    # Intersection: Sum of absolute element-wise product
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    
    # Union: Sum of individual absolute values, minus intersection
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    
    # Jaccard Index (with smoothing to avoid division by zero)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    
    # Jaccard Distance Loss: (1 - IoU) * smooth (scaled for stability)
    return (1 - jac) * smooth