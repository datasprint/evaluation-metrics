import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.nonparametric import kernels
from statsmodels.nonparametric import bandwidths

feature = "days_since_birth"

kernel_switch = dict(gau=kernels.Gaussian, epa=kernels.Epanechnikov,
                    uni=kernels.Uniform, tri=kernels.Triangular,
                    biw=kernels.Biweight, triw=kernels.Triweight,
                    cos=kernels.Cosine, cos2=kernels.Cosine2)
bw = bandwidths.select_bandwidth(data_dummies[feature].astype('double'), "normal_reference", kernel_switch["gau"]())


def kde_func(series, bw, additional=[]):
  kde = sm.nonparametric.KDEUnivariate(series)
  kde.fit(bw=bw) # Estimate the densities

  kde.support
  kde.density
  fig = plt.figure(figsize=(12, 5))
  ax = fig.add_subplot(111)

  # Plot the histrogram
  ax.hist(series, bins=20, density=True, label='Histogram from samples',
          zorder=5, edgecolor='k', alpha=0.5)

  # Plot the KDE as fitted using the default arguments
  ax.plot(kde.support, kde.density, lw=3, label='KDE from samples', zorder=10)

  if len(additional)>0:
    for r in additional:
      ax.plot(kde.support, r, lw=3, label='KDE other: {}'.format(r), zorder=10)

  return ax, kde.support, kde.density, kde 

def rescale_interpolate(df_org, df_gen, feat):
  max_target = df_org[feat].astype(int).max()
  min_target = df_org[feat].astype(int).min()
  change = df_gen[feat]
  #change = df_org["Age"].sample(int(len(df_org)*0.5))
  scaled_array = np.interp(change, (change.min(), change.max()), (min_target, max_target))
  return scaled_array


pl, support_org, density_org, kde_org = kde_func(data_dummies[feature].astype('double'), bw)


scaled_array = rescale_interpolate(data_dummies, samples_dummies, feature)
pl, support_gen, density_gen, kde_gen = kde_func(scaled_array, bw, [density_org] )


def dict_curve(matrix_org_s,matrix_gen_s):
  c_dict = {}
  c_dict["Curve Length Difference"] = round(curve_length_measure(matrix_org_s, matrix_gen_s),5)
  c_dict["Partial Curve Mapping"] = round(pcm(matrix_gen_s, matrix_org_s),5)
  c_dict["Discrete Frechet Distance"] = round(frechet_dist(matrix_gen_s, matrix_org_s),5)
  c_dict["Dynamic Time Warping"] = round(dtw(matrix_gen_s, matrix_org_s)[0],5)
  c_dict["Area Between Curves"] = round(area_between_two_curves(matrix_org_s, matrix_gen_s),5)
  return c_dict

from scipy.spatial import distance, minkowski_distance


def area_u_c(matrix_org_s,matrix_gen_s,):

  x = matrix_org_s[:, 0]
  z = matrix_org_s[:, 1]-matrix_gen_s[:, 1]
  dx = x[1:] - x[:-1]
  cross_test = np.sign(z[:-1] * z[1:])

  x_intersect = x[:-1] - dx / (z[1:] - z[:-1]) * z[:-1]
  dx_intersect = - dx / (z[1:] - z[:-1]) * z[:-1]

  areas_pos = abs(z[:-1] + z[1:]) * 0.5 * dx # signs of both z are same
  areas_neg = 0.5 * dx_intersect * abs(z[:-1]) + 0.5 * (dx - dx_intersect) * abs(z[1:])

  areas = np.where(cross_test < 0, areas_neg, areas_pos)
  total_area = np.sum(areas)
  return total_area



def poly_area(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def is_simple_quad(ab, bc, cd, da):
    #   Compute all four cross products
    temp0 = np.cross(ab, bc)
    temp1 = np.cross(bc, cd)
    temp2 = np.cross(cd, da)
    temp3 = np.cross(da, ab)
    cross = np.array([temp0, temp1, temp2, temp3])
    #   See that cross products are greater than or equal to zero
    crossTF = cross >= 0
    #   if the cross products are majority false, re compute the cross products
    #   Because they don't necessarily need to lie in the same 'Z' direction
    if sum(crossTF) <= 1:
        crossTF = cross <= 0
    if sum(crossTF) > 2:
        return True
    else:
        return False


def makeQuad(x, y):
    AB = [x[1]-x[0], y[1]-y[0]]
    BC = [x[2]-x[1], y[2]-y[1]]
    CD = [x[3]-x[2], y[3]-y[2]]
    DA = [x[0]-x[3], y[0]-y[3]]

    isQuad = is_simple_quad(AB, BC, CD, DA)

    if isQuad is False:
        # attempt to rearrange the first two points
        x[1], x[0] = x[0], x[1]
        y[1], y[0] = y[0], y[1]
        AB = [x[1]-x[0], y[1]-y[0]]
        BC = [x[2]-x[1], y[2]-y[1]]
        CD = [x[3]-x[2], y[3]-y[2]]
        DA = [x[0]-x[3], y[0]-y[3]]

        isQuad = is_simple_quad(AB, BC, CD, DA)

        if isQuad is False:
            # place the second and first points back where they were, and
            # swap the second and third points
            x[2], x[0], x[1] = x[0], x[1], x[2]
            y[2], y[0], y[1] = y[0], y[1], y[2]
            AB = [x[1]-x[0], y[1]-y[0]]
            BC = [x[2]-x[1], y[2]-y[1]]
            CD = [x[3]-x[2], y[3]-y[2]]
            DA = [x[0]-x[3], y[0]-y[3]]

            isQuad = is_simple_quad(AB, BC, CD, DA)

    # calculate the area via shoelace formula
    area = poly_area(x, y)
    return area


def get_arc_length(dataset):
    #   split the dataset into two discrete datasets, each of length m-1
    m = len(dataset)
    a = dataset[0:m-1, :]
    b = dataset[1:m, :]
    #   use scipy.spatial to compute the euclidean distance
    dataDistance = distance.cdist(a, b, 'euclidean')
    #   this returns a matrix of the euclidean distance between all points
    #   the arc length is simply the sum of the diagonal of this matrix
    arcLengths = np.diagonal(dataDistance)
    arcLength = sum(arcLengths)
    return arcLength, arcLengths


def area_between_two_curves(exp_data, num_data):
    n_exp = len(exp_data)
    n_num = len(num_data)

    # the length of exp_data must be larger than the length of num_data
    if n_exp < n_num:
        temp = num_data.copy()
        num_data = exp_data.copy()
        exp_data = temp.copy()
        n_exp = len(exp_data)
        n_num = len(num_data)

    # get the arc length data of the curves
    # arcexp_data, _ = get_arc_length(exp_data)
    _, arcsnum_data = get_arc_length(num_data)

    # let's find the largest gap between point the num_data, and then
    # linearally interpolate between these points such that the num_data
    # becomes the same length as the exp_data
    for i in range(0, n_exp-n_num):
        a = num_data[0:n_num-1, 0]
        b = num_data[1:n_num, 0]
        nIndex = np.argmax(arcsnum_data)
        newX = (b[nIndex] + a[nIndex])/2.0
        #   the interpolation model messes up if x2 < x1 so we do a quick check
        if a[nIndex] < b[nIndex]:
            newY = np.interp(newX, [a[nIndex], b[nIndex]],
                             [num_data[nIndex, 1], num_data[nIndex+1, 1]])
        else:
            newY = np.interp(newX, [b[nIndex], a[nIndex]],
                             [num_data[nIndex+1, 1], num_data[nIndex, 1]])
        num_data = np.insert(num_data, nIndex+1, newX, axis=0)
        num_data[nIndex+1, 1] = newY

        _, arcsnum_data = get_arc_length(num_data)
        n_num = len(num_data)

    # Calculate the quadrilateral area, by looping through all of the quads
    area = []
    for i in range(1, n_exp):
        tempX = [exp_data[i-1, 0], exp_data[i, 0], num_data[i, 0],
                 num_data[i-1, 0]]
        tempY = [exp_data[i-1, 1], exp_data[i, 1], num_data[i, 1],
                 num_data[i-1, 1]]
        area.append(makeQuad(tempX, tempY))
    return np.sum(area)


def get_length(x, y):
    n = len(x)
    xmax = np.max(np.abs(x))
    ymax = np.max(np.abs(y))

    # if your max x or y value is zero... you'll get np.inf
    # as your curve length based measure
    if xmax == 0:
        xmax = 1e-15
    if ymax == 0:
        ymax = 1e-15

    le = np.zeros(n)
    le[0] = 0.0
    l_sum = np.zeros(n)
    l_sum[0] = 0.0
    for i in range(0, n-1):
        le[i+1] = np.sqrt((((x[i+1]-x[i])/xmax)**2)+(((y[i+1]-y[i])/ymax)**2))
        l_sum[i+1] = l_sum[i]+le[i+1]
    return le, np.sum(le), l_sum


def curve_length_measure(exp_data, num_data):
    x_e = exp_data[:, 0]
    y_e = exp_data[:, 1]
    x_c = num_data[:, 0]
    y_c = num_data[:, 1]

    _, le_nj, le_sum = get_length(x_e, y_e)
    _, lc_nj, lc_sum = get_length(x_c, y_c)

    xmean = np.mean(x_e)
    ymean = np.mean(y_e)

    n = len(x_e)

    r_sq = np.zeros(n)
    for i in range(0, n):
        lieq = le_sum[i]*(lc_nj/le_nj)
        xtemp = np.interp(lieq, lc_sum, x_c)
        ytemp = np.interp(lieq, lc_sum, y_c)

        r_sq[i] = np.log(1.0 + (np.abs(xtemp-x_e[i])/xmean))**2 + \
            np.log(1.0 + (np.abs(ytemp-y_e[i])/ymean))**2
    return np.sqrt(np.sum(r_sq))


def frechet_dist(exp_data, num_data, p=2):
    n = len(exp_data)
    m = len(num_data)
    ca = np.ones((n, m))
    ca = np.multiply(ca, -1)
    ca[0, 0] = minkowski_distance(exp_data[0], num_data[0], p=p)
    for i in range(1, n):
        ca[i, 0] = max(ca[i-1, 0], minkowski_distance(exp_data[i], num_data[0],
                                                      p=p))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], minkowski_distance(exp_data[0], num_data[j],
                                                      p=p))
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]),
                           minkowski_distance(exp_data[i], num_data[j], p=p))
    return ca[n-1, m-1]


def normalizeTwoCurves(x, y, w, z):

    minX = np.min(x)
    maxX = np.max(x)
    minY = np.min(y)
    maxY = np.max(y)

    xi = (x - minX) / (maxX - minX)
    eta = (y - minY) / (maxY - minY)
    xiP = (w - minX) / (maxX - minX)
    etaP = (z - minY) / (maxY - minY)
    return xi, eta, xiP, etaP


def pcm(exp_data, num_data):
    # normalize the curves to the experimental data
    xi1, eta1, xi2, eta2 = normalizeTwoCurves(exp_data[:, 0], exp_data[:, 1],
                                              num_data[:, 0], num_data[:, 1])
    # compute the arc lengths of each curve
    le, le_nj, le_sum = get_length(xi1, eta1)
    lc, lc_nj, lc_sum = get_length(xi2, eta2)
    # scale each segment to the total polygon length
    le = le / le_nj
    le_sum = le_sum / le_nj
    lc = lc / lc_nj
    lc_sum = lc_sum / lc_nj
    # right now exp_data is curve a, and num_data is curve b
    # make sure a is shorter than a', if not swap the defintion
    if lc_nj > le_nj:
        # compute the arc lengths of each curve
        le, le_nj, le_sum = get_length(xi2, eta2)
        lc, lc_nj, lc_sum = get_length(xi1, eta1)
        # scale each segment to the total polygon length
        le = le / le_nj
        le_sum = le_sum / le_nj
        lc = lc / lc_nj
        lc_sum = lc_sum / lc_nj
        # swap xi1, eta1 with xi2, eta2
        xi1OLD = xi1.copy()
        eta1OLD = eta1.copy()
        xi1 = xi2.copy()
        eta1 = eta2.copy()
        xi2 = xi1OLD.copy()
        eta2 = eta1OLD.copy()

    n_sum = len(le_sum)

    min_offset = 0.0
    max_offset = le_nj - lc_nj

    # make sure the curves aren't the same length
    # if they are the same length, don't loop 200 times
    if min_offset == max_offset:
        offsets = [min_offset]
        pcm_dists = np.zeros(1)
    else:
        offsets = np.linspace(min_offset, max_offset, 200)
        pcm_dists = np.zeros(200)

    for i, offset in enumerate(offsets):
        # create linear interpolation model for num_data based on arc length
        # evaluate linear interpolation model based on xi and eta of exp data
        xitemp = np.interp(le_sum+offset, lc_sum, xi2)
        etatemp = np.interp(le_sum+offset, lc_sum, eta2)

        d = np.sqrt((eta1-etatemp)**2 + (xi1-xitemp)**2)
        d1 = d[:-1]

        d2 = d[1:n_sum]

        v = 0.5*(d1+d2)*le_sum[1:n_sum]
        pcm_dists[i] = np.sum(v)
    return np.min(pcm_dists)


def dtw(exp_data, num_data, metric='euclidean', **kwargs):
    c = distance.cdist(exp_data, num_data, metric=metric, **kwargs)

    d = np.zeros(c.shape)
    d[0, 0] = c[0, 0]
    n, m = c.shape
    for i in range(1, n):
        d[i, 0] = d[i-1, 0] + c[i, 0]
    for j in range(1, m):
        d[0, j] = d[0, j-1] + c[0, j]
    for i in range(1, n):
        for j in range(1, m):
            d[i, j] = c[i, j] + min((d[i-1, j], d[i, j-1], d[i-1, j-1]))
    return d[-1, -1], d


def dtw_path(d):
    path = []
    i, j = d.shape
    i = i - 1
    j = j - 1
    # back propagation starts from the last point,
    # and ends at d[0, 0]
    path.append((i, j))
    while i > 0 or j > 0:
        if i == 0:
            j = j - 1
        elif j == 0:
            i = i - 1
        else:
            temp_step = min([d[i-1, j], d[i, j-1], d[i-1, j-1]])
            if d[i-1, j] == temp_step:
                i = i - 1
            elif d[i, j-1] == temp_step:
                j = j - 1
            else:
                i = i - 1
                j = j - 1
        path.append((i, j))
    path = np.array(path)
    # reverse the order of path, such that it starts with [0, 0]
    return path[::-1]


matrix_org = np.array([support_org,density_org]).T
matrix_gen = np.array([support_gen,density_gen]).T

dict_curve(matrix_org, matrix_gen)

