#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2017 Paul Brodersen <paulbrodersen+vizout@gmail.com>

# Author: Paul Brodersen <paulbrodersen+vizout@gmail.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# --------------------------------------------------------------------------------
# definitions
# --------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="white")
import pandas, pylab

def test():
    from scipy.stats import multivariate_normal

    # create some Gaussian test data
    n = 1000 # number of data points
    d = 4 # dimensionality of data
    mu = np.zeros((d)) # mean
    sigma = np.random.rand(d,d) # std
    sigma = np.dot(sigma.transpose(), sigma) # make positive semi-definite
    regular_samples = multivariate_normal(mu, sigma).rvs(n)

    # add some outliers
    m = n / 20
    sigma *= 10
    outliers = multivariate_normal(mu, sigma).rvs(m)

    data_points = np.r_[regular_samples, outliers]

    # re-express the data points in terms of the first 3 principal components;
    # with only 4 data dimensions, plotting all 4 would not have been a problem,
    # but such a dimensionality reduction is useful for very high dimensional data sets
    reduced_points = reduce_dimensionality(data_points, ndim=3, method='pca', whiten=True)

    # plot the data points along the first 3 principal components;
    selected_indices = main(reduced_points)

    # now select outliers by clicking on them
    return selected_indices


def main(data_points):
    """
    Provides a graphical interface to select outliers in high dimensional
    data sets interactively.

    Arguments:
    ----------
        data_points: (N samples, D dimensions) numpy ndarray
            data points

    Returns:
    --------
        outlier_indices: (M, ) numpy ndarray
            indices into the original data array corresponding to the selected outliers

    Example:
    --------

    import vizout
    from scipy.stats import multivariate_normal

    # create some Gaussian test data
    n = 1000 # number of data points
    d = 4 # dimensionality of data
    mu = np.zeros((d)) # mean
    sigma = np.random.rand(d,d) # std
    sigma = np.dot(sigma.transpose(), sigma) # make positive semi-definite
    regular_samples = multivariate_normal(mu, sigma).rvs(n)

    # add some outliers
    m = n / 20
    sigma *= 10
    outliers = multivariate_normal(mu, sigma).rvs(m)

    data_points = np.r_[regular_samples, outliers]

    # re-express the data points in terms of the first 3 principal components;
    # with only 4 data dimensions, plotting all 4 would not have been a problem,
    # but such a dimensionality reduction is useful for very high dimensional data sets
    reduced_points = vizout.reduce_dimensionality(data_points, ndim=3, method='pca', whiten=True)

    # plot the data points along the first 3 principal components;
    selected_indices = vizout.main(reduced_points)

    # now select outliers by clicking on them
    """

    # set up canvas
    df = pandas.DataFrame(data_points) # bloody seaborn needs pandas data frames
    grid = sns.PairGrid(df, diag_sharey=False)
    grid.map_lower(sns.kdeplot, cmap="Blues_d")
    grid.map_upper(plt.scatter, s=1.)
    grid.map_diag(sns.kdeplot, lw=2.)

    # link scatter subplots
    all_annotators = []
    data_point_labels = range(len(data_points))
    for ii in range(grid.axes.shape[0]):
        for jj in range(grid.axes.shape[1]):
            if jj > ii: # upper triangle
                annotator = _label_points(data_points[:,jj],
                                          data_points[:,ii],
                                          data_point_labels,
                                          ax=grid.axes[ii,jj])
                pylab.connect('button_press_event', annotator)
                all_annotators.append(annotator)
    _link_annotators(all_annotators)

    # handle callback
    plt.show()
    outlier_indices = set()
    for annotator in all_annotators:
        for (x, y, label) in annotator.drawn_annotations.iterkeys():
            outlier_indices.add(label)

    return outlier_indices


def reduce_dimensionality(array, ndim, method='pca', whiten=True, verbose=False):
    """
    Arguments:
    ----------
        array: (n samples, d dimensions) numpy.ndarray
            data array
        ndim: int
            number of target dimensions
        method: str
            method to reduce dimensionality; one of
                'pca' (only one implemented so far)
        whiten: bool
            whether to whiten the data before dimensionality reduction

    Returns:
    --------
        reduced_array: (n samples, ndim dimensions) numpy.ndarray
            data array of reduced dimensionality
    """

    # dependencies not needed if all dimensions are visualised
    # -> import here
    from sklearn.preprocessing import Imputer
    from sklearn.decomposition import PCA

    # rescale data
    array -= np.nanmin(array, axis=0)[None, :]
    array /= np.nanmax(array, axis=0)[None, :]

    # fill in missing values
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp.fit(array)
    array = imp.transform(array)

    # run method and transform data
    if method is 'pca':
        pca = PCA(n_components=ndim, whiten=whiten)
        pca.fit(array)
        reduced_array = pca.transform(array)

        if verbose:
            print "% variance explained (cumulative):"
            evr = pca.explained_variance_ratio_ * 100
            for ii, (v, cv) in enumerate(zip(evr, np.cumsum(evr))):
                print "pc #{}: {:2.1f} ({:2.1f}) %".format(ii+1, v, cv)

    else:
        raise ValueError("Method one of ['pca']; current value: {}".format(method))

    return reduced_array


class _label_points(object):
    """
    Callback for matplotlib to display an annotation when points are
    clicked on. The point which is closest to the click is identified.
    If there are several points within 'tol' from the selected point,
    all are marked.

    Register this function like this:

    fig, ax = plt.subplots(1,1)
    ax.scatter(xdata, ydata)
    a = Annotator(xdata, ydata, labels, ax=ax, tol=0.)
    connect('button_press_event', a)

    @reference:
    http://scipy-cookbook.readthedocs.io/items/Matplotlib_Interactive_Plotting.html

    """

    def __init__(self, xdata, ydata, labels, ax=None, tol=None):
        self.xdata= xdata
        self.ydata = ydata
        self.labels = np.array(labels)
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        if tol is None:
            dx = np.diff(self.ax.get_xlim())
            dy = np.diff(self.ax.get_ylim())
            tol = np.sqrt((0.01*dx)**2 + (0.01*dy)**2)
        self.tol = tol
        self.drawn_annotations = {}
        self.links = []

    def get_distance(self, x, y):
        return np.sqrt((self.xdata - x)**2 + (self.ydata - y)**2)

    def __call__(self, event):

        if event.inaxes is self.ax:
            # find nearest point
            distances = self.get_distance(event.xdata, event.ydata)
            idx = np.argmin(distances)

            # label point and all points that are tol away
            # this is useful if there are many points on top of each other
            distances = self.get_distance(self.xdata[idx], self.ydata[idx])
            idx, = np.where(distances <= self.tol)

            for ii in idx:
                x, y, label = self.xdata[ii], self.ydata[ii], self.labels[ii]
                print label

                self.draw_label(event.inaxes, x, y, label)
                for link in self.links:
                    link.annotate_point(label)

            # update figure
            self.ax.figure.canvas.draw_idle()

    def draw_label(self, ax, x, y, label):
        """
        Draw the annotation on the plot.
        """
        if (x, y, label) in self.drawn_annotations:
            # markers = self.drawnAnnotations[(x,y,label)]
            markers = self.drawn_annotations.pop((x, y, label))
            for m in markers:
                m.set_visible(not m.get_visible())
            # self.ax.figure.canvas.draw_idle() <- much faster to draw after updating all axes
        else:
            t = ax.text(x, y, "%s" % (label),)
            m = ax.scatter([x], [y], marker='d', c='r', zorder=100)
            self.drawn_annotations[(x, y, label)] = (t, m)
            # self.ax.figure.canvas.draw_idle()

    def annotate_point(self, label):
        hits, = np.where(self.labels == label)
        for ii in hits:
            self.draw_label(self.ax, self.xdata[ii], self.ydata[ii], self.labels[ii])

def _link_annotators(annotators):
    """
    Sets up synchronisation of data point labels across subplots.

    @reference:
    http://scipy-cookbook.readthedocs.io/items/Matplotlib_Interactive_Plotting.html

    """
    for ii, annotator in enumerate(annotators):
        all_but_self = annotators[:ii]+annotators[ii+1:]
        annotator.links.extend(all_but_self)


# --------------------------------------------------------------------------------
# script
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    test()
