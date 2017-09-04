__author__ = 'sibirrer'

import lenstronomy.util as Util
import astrofunc.util as Util_astrofunc
import pytest
import numpy.testing as npt


def test_findOverlap():
    x_mins = [0,1,0]
    y_mins = [1,2,1]
    values = [0.0001,1,0.001]
    deltapix = 1
    x_mins, y_mins, values = Util.findOverlap(x_mins, y_mins, values, deltapix)
    assert x_mins == 0
    assert y_mins == 1
    assert values == 0.0001


def test_coordInImage():
    x_coord = [100,20,-10]
    y_coord = [0,-30,5]
    numPix = 50
    deltapix = 1
    x_result, y_result = Util.coordInImage(x_coord, y_coord, numPix, deltapix)
    assert x_result == -10
    assert y_result == 5


def test_rebin_coord_transform():
    x_grid, y_grid, x_0, y_0, ra_0, dec_0, Mpix2coord, Mcoord2pix = Util_astrofunc.make_grid_with_coordtransform(numPix=3, deltapix=0.03, subgrid_res=1)
    x_grid, y_grid, x_0_re, y_0_re, ra_0_re, dec_0_re, Mpix2coord_re, Mcoord2pix_re = Util_astrofunc.make_grid_with_coordtransform(numPix=1, deltapix=0.09, subgrid_res=1)

    ra_0_resized, dec_0_resized, x_0_resized, y_0_resized, Mpix2coord_resized, Mcoord2pix_resized = Util.rebin_coord_transform(3, ra_0, dec_0, x_0, y_0, Mpix2coord, Mcoord2pix)
    assert ra_0_resized == ra_0_re
    assert dec_0_resized == dec_0_re
    assert x_0_resized == x_0_re
    assert y_0_resized == y_0_re
    npt.assert_almost_equal(Mcoord2pix_resized[0][0], Mcoord2pix_re[0][0], decimal=8)
    npt.assert_almost_equal(Mpix2coord_re[0][0], Mpix2coord_resized[0][0], decimal=8)


    x_grid, y_grid, x_0, y_0, ra_0, dec_0, Mpix2coord, Mcoord2pix = Util_astrofunc.make_grid_with_coordtransform(numPix=100, deltapix=0.05, subgrid_res=1)
    x_grid, y_grid, x_0_re, y_0_re, ra_0_re, dec_0_re, Mpix2coord_re, Mcoord2pix_re = Util_astrofunc.make_grid_with_coordtransform(numPix=50, deltapix=0.1, subgrid_res=1)

    ra_0_resized, dec_0_resized, x_0_resized, y_0_resized, Mpix2coord_resized, Mcoord2pix_resized = Util.rebin_coord_transform(2, ra_0, dec_0, x_0, y_0, Mpix2coord, Mcoord2pix)
    print(ra_0_re, dec_0_re, x_0_re, y_0_re, Mpix2coord_re, Mcoord2pix_re)
    print(ra_0_resized, dec_0_resized, x_0_resized, y_0_resized, Mpix2coord_resized, Mcoord2pix_resized)
    assert ra_0_resized == ra_0_re
    assert dec_0_resized == dec_0_re
    assert x_0_resized == x_0_re
    assert y_0_resized == y_0_re
    npt.assert_almost_equal(Mcoord2pix_resized[0][0], Mcoord2pix_re[0][0], decimal=8)
    npt.assert_almost_equal(Mpix2coord_re[0][0], Mpix2coord_resized[0][0], decimal=8)

    x_grid, y_grid, x_0, y_0, ra_0, dec_0, Mpix2coord, Mcoord2pix = Util_astrofunc.make_grid_with_coordtransform(numPix=99, deltapix=0.1, subgrid_res=1)
    x_grid, y_grid, x_0_re, y_0_re, ra_0_re, dec_0_re, Mpix2coord_re, Mcoord2pix_re = Util_astrofunc.make_grid_with_coordtransform(numPix=33, deltapix=0.3, subgrid_res=1)

    ra_0_resized, dec_0_resized, x_0_resized, y_0_resized, Mpix2coord_resized, Mcoord2pix_resized = Util.rebin_coord_transform(3, ra_0, dec_0, x_0, y_0, Mpix2coord, Mcoord2pix)
    print(ra_0_re, dec_0_re, x_0_re, y_0_re, Mpix2coord_re, Mcoord2pix_re)
    print(ra_0_resized, dec_0_resized, x_0_resized, y_0_resized, Mpix2coord_resized, Mcoord2pix_resized)
    assert ra_0_resized == ra_0_re
    assert dec_0_resized == dec_0_re
    npt.assert_almost_equal(x_0_resized, x_0_re, decimal=8)
    npt.assert_almost_equal(y_0_resized, y_0_re, decimal=8)
    npt.assert_almost_equal(Mcoord2pix_resized[0][0], Mcoord2pix_re[0][0], decimal=8)
    npt.assert_almost_equal(Mpix2coord_re[0][0], Mpix2coord_resized[0][0], decimal=8)

if __name__ == '__main__':
    pytest.main()