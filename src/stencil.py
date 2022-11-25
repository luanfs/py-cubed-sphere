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

    if simulation.flux_method_name == 'PPM':
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

        ax[:,:,:] = ax[:,:,:]/12.0

    elif simulation.flux_method_name == 'PPM_hybrid':
        ax[0, upositive] =  2.0 - cx[upositive] - cx2[upositive]
        ax[0, unegative] =  0.0

        ax[1, upositive] = -13.0 -      cx[upositive] + 14.0*cx2[upositive]
        ax[1, unegative] =  -3.0 +  4.0*cx[unegative] -      cx2[unegative]

        ax[2, upositive] =  47.0 + 39.0*cx[upositive] - 26.0*cx2[upositive]
        ax[2, unegative] =  27.0 - 41.0*cx[unegative] + 14.0*cx2[unegative]

        ax[3, upositive] =  27.0 - 41.0*cx[upositive] + 14.0*cx2[upositive]
        ax[3, unegative] =  47.0 + 39.0*cx[unegative] - 26.0*cx2[unegative]

        ax[4, upositive] =  -3.0 +  4.0*cx[upositive] -      cx2[upositive]
        ax[4, unegative] = -13.0 -      cx[unegative] + 14.0*cx2[unegative]

        ax[5, upositive] =  0.0
        ax[5, unegative] =  2.0 - cx[unegative] - cx2[unegative]

        ax[:,:,:] = ax[:,:,:]/60.0

####################################################################################
# Compute the 1d PPM flux stencil coefficients
####################################################################################
def flux_ppm_y_stencil_coefficients(v_edges, ay, cy, cy2, simulation):
    # Stencil coefficients
    vpositive = v_edges>=0
    vnegative = ~vpositive

    if simulation.flux_method_name == 'PPM':
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

        ay[:,:,:] = ay[:,:,:]/12.0

    elif simulation.flux_method_name == 'PPM_hybrid':
        ay[0, vpositive] =  2.0 - cy[vpositive] - cy2[vpositive]
        ay[0, vnegative] =  0.0

        ay[1, vpositive] = -13.0 -      cy[vpositive] + 14.0*cy2[vpositive]
        ay[1, vnegative] =  -3.0 +  4.0*cy[vnegative] -      cy2[vnegative]

        ay[2, vpositive] =  47.0 + 39.0*cy[vpositive] - 26.0*cy2[vpositive]
        ay[2, vnegative] =  27.0 - 41.0*cy[vnegative] + 14.0*cy2[vnegative]

        ay[3, vpositive] =  27.0 - 41.0*cy[vpositive] + 14.0*cy2[vpositive]
        ay[3, vnegative] =  47.0 + 39.0*cy[vnegative] - 26.0*cy2[vnegative]

        ay[4, vpositive] =  -3.0 +  4.0*cy[vpositive] -      cy2[vpositive]
        ay[4, vnegative] = -13.0 -      cy[vnegative] + 14.0*cy2[vnegative]

        ay[5, vpositive] =  0.0
        ay[5, vnegative] =  2.0 - cy[vnegative] - cy2[vnegative]

        ay[:,:,:] = ay[:,:,:]/60.0
