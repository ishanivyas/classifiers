# Exploring the Limits of Classifiers in a Hostile Environment

The purpose of this project is to evaluate a representative set of classifiers in as pessimistic a way as possible. Therefore, the perspective chosen is white-box adversarial. The adversary can see detailed statistics for the classifier under evaluation and use that information to construct deceptive inputs. The adversary starts by introducing small changes to well-classified inputs and then proceeds to increase the size of the change until the classifier picks the wrong class.

>"Effectiveness" in increasing order of strength:
  1. The classifier still outputs the correct class for a set of features it classified correctly in the past.
  2. The classifier outputs the wrong class for a set of features it used to classify correctly.
  3. The image is not noticeably different from the original input.
  4. The classifier outputs the class we are trying to get it to output.

 ## Strategy 2:
 - Find the top-confused other-class that has *THIS* class as *ITS* top-confused class.
 - The idea here is that these 2 classes are easy to confuse, so we don't have to push very hard to tip one into the other using the other's features.
 - Consider the deceptor weighting that normalizes row sums to 1, combines reciprocal relationships, and scores the deceptor as the max of the reciprocal-relationship and the one-sided relationship:
 - UnitD = axis-divide(D, axis-max(axis-sum(D,row),1)) # Normalized confusion/deceptor rows (sum of row == 1). Careful to avoid divide-by-zero!
 - D = axis-argmax(np.max(np.sqrt(UnitD * UnitD.T), UnitD)) # sqrt() over the interval [0,1) should make the comparison of the product to the un-multiplied UnitD more fair.  ... which could represent a decent compromise between Strategy 1 and Strategy 2.

## Strategy 3:
- Find the top 2 classes it recognizes, "a", and "b".
- Try to get everything to be classified as the top class...
- EXCEPT the top class which we should try to get classified as the 2nd class.
- D is a matrix composed entirely of deveptive images to mix with X

## Strategy 4:
- Use the the least-often-confused non-zero other-class for each class.
- The idea here is that it will be easier to find a small set of archetypes that confuse the classifier.
- These are potentially the outliers of the other class that are most like THIS class.
