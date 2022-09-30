####################################################################################
#
# Module for 1D FV flux stencils
# Luan da Fonseca Santos - September 2022
# (luan.santos@usp.br)
####################################################################################

####################################################################################
# Compute the 1d PPM flux stencil coefficients
####################################################################################
def flux_ppm_x_stencil_coefficients(u_edges, ax, cx, cx2, simulation):
    # Stencil coefficients
    upositive = u_edges>=0
    unegative = ~upositive

    ax[0, upositive] =  cx[upositive] - cx2[upositive]
    ax[0, unegative] =  0.0

    ax[1, upositive] = -1.0 - 5.0*cx[upositive] + 6.0*cx2[upositive]
    ax[1, unegative] = -1.0 + 2.0*cx[unegative] - cx2[unegative] 

    ax[2, upositive] =  7.0 + 15.0*cx[upositive] - 10.0*cx2[upositive]
    ax[2, unegative] =  7.0 - 13.0*cx[unegative] + 6.0*cx2[unegative] 

    ax[3, upositive] =  7.0 - 13.0*cx[upositive] + 6.0*cx2[upositive]
    ax[3, unegative] =  7.0 + 15.0*cx[unegative] - 10.0*cx2[unegative] 

    ax[4, upositive] = -1.0 + 2.0*cx[upositive] - cx2[upositive]
    ax[4, unegative] = -1.0 - 5.0*cx[unegative] + 6.0*cx2[unegative] 

    ax[5, upositive] =  0.0
    ax[5, unegative] =  cx[unegative] - cx2[unegative] 

####################################################################################
# Compute the 1d PPM flux stencil coefficients
####################################################################################
def flux_ppm_y_stencil_coefficients(v_edges, ay, cy, cy2, simulation):
    # Stencil coefficients
    vpositive = v_edges>=0
    vnegative = ~vpositive

    ay[0, vpositive] =  cy[vpositive] - cy2[vpositive]
    ay[0, vnegative] =  0.0

    ay[1, vpositive] = -1.0 - 5.0*cy[vpositive] + 6.0*cy2[vpositive]
    ay[1, vnegative] = -1.0 + 2.0*cy[vnegative] - cy2[vnegative] 

    ay[2, vpositive] =  7.0 + 15.0*cy[vpositive] - 10.0*cy2[vpositive]
    ay[2, vnegative] =  7.0 - 13.0*cy[vnegative] + 6.0*cy2[vnegative] 

    ay[3, vpositive] =  7.0 - 13.0*cy[vpositive] + 6.0*cy2[vpositive]
    ay[3, vnegative] =  7.0 + 15.0*cy[vnegative] - 10.0*cy2[vnegative] 

    ay[4, vpositive] = -1.0 + 2.0*cy[vpositive] - cy2[vpositive]
    ay[4, vnegative] = -1.0 - 5.0*cy[vnegative] + 6.0*cy2[vnegative] 

    ay[5, vpositive] =  0.0
    ay[5, vnegative] =  cy[vnegative] - cy2[vnegative]
