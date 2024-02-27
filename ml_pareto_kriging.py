import numpy as np
from matplotlib import pyplot as plt
import gstools as gs
import lhsmdu as lhs
import random as rd

#kriging_type "simple" / "ordinary" /"universal_linear" / "universal_quadratic"
def calculateKrigingModel(kriging_type, model_in, x_in, y_in, z_in):
    krig = {}
    if kriging_type == "simple":
        krig = gs.krige.Simple(model_in, cond_pos=(x_in, y_in), cond_val=z_in)
    elif kriging_type == "ordinary":
        krig = gs.krige.Ordinary(model_in, cond_pos=(x_in, y_in), cond_val=z_in)
    elif kriging_type == "universal_linear":
        krig = gs.krige.Universal(model_in, cond_pos=(x_in, y_in), cond_val=z_in, drift_functions="linear")
    elif kriging_type == "universal_quadratic":
        krig = gs.krige.Universal(model_in, cond_pos=(x_in, y_in), cond_val=z_in, drift_functions="quadratic")
    elif kriging_type == "detrended":
        def trend(x):
            """Trend muss manuell angepasst werden."""
            return x * 0.1 + 1
        print("Nicht getestet, k.a. ob's funktioniert")
        krig = gs.krige.Detrended(model_in, cond_pos=(x_in, y_in), cond_val=z_in, trend=trend)
    else:
        print("Überprüfe kriging-type")

    return krig

def plotKrigModel(krig_model, min_x, max_x, min_y, max_y, x_p, y_p, fig, ax1_in, ax2_in):
    n_points_to_plot = 100
    x, y = np.meshgrid(np.linspace(min_x, max_x, n_points_to_plot), np.linspace(min_y, max_y, n_points_to_plot))

    krige_field, krige_var = krig_model((x, y), return_var=True, only_mean=False)

    krige_field_reshaped = np.reshape(krige_field, (n_points_to_plot, n_points_to_plot))

    fig.suptitle('Plot des Kriging-Models')

    z_min = np.min(krige_field)
    z_max = np.max(krige_field)

    levels = np.linspace(z_min, z_max, n_points_to_plot + 1)

    # plot predicted function
    contourf_ = ax1_in.contourf(x, y, krige_field_reshaped, levels, vmin=z_min, vmax=z_max, cmap='viridis')
    ax1_in.title.set_text('Predicted')
    fig.colorbar(contourf_, ax=ax1_in)
    ax1_in.scatter(x_p, y_p, color="red")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    z_std_arr1d = np.sqrt(krige_var)
    z_std = np.reshape(z_std_arr1d, (n_points_to_plot, n_points_to_plot))

    z_min = np.min(z_std_arr1d)
    z_max = np.max(z_std_arr1d)
    levels = np.linspace(z_min, z_max, n_points_to_plot + 1)
    contourf_ = ax2_in.contourf(x, y, z_std, levels, vmin=z_min, vmax=z_max, cmap='viridis')
    ax2_in.title.set_text('Kriging Std.Abweichung-> mean(): %.3f' % np.mean(z_std_arr1d))
    plt.colorbar(contourf_, ax=ax2_in)
    ax2_in.scatter(x_p, y_p, color="red")

    fig.set_size_inches(15, 5)

#################  Es steht ein skript bereit, mit dem ihr den MSE in Abhängigkeit von der Anzahl der Startpunkte untersuchen könnt ##########
## ihr könnt die obere oder beliebige funktion damit untersuchen ##
## skript führt den fit mit allen punkten durch, zusätzlich mit aufsplittung in startpunkte + größter SE

def analyseFunction(f1, n_points_to_fit, n_runs, init_faction, fig, ax1, ax2, ax3, ax4, krig_type = "universal_quadratic", start_points = "CenteredRandomGridCombinedPointsWithEdges"):

    min_x, max_x, min_y, max_y = f1.getRange()

    n_points_to_plot = 100
    x, y = np.meshgrid(np.linspace(min_x, max_x, n_points_to_plot), np.linspace(min_y, max_y, n_points_to_plot))
    z = f1.getValue(x, y)

    model_mse = []
    model_mse_std = []

    best_krige = {}
    best_cov_model = {}
    best_bin_center = {}
    best_gamma = {}
    best_mse_all = np.finfo(np.float64).max
    best_x = {}
    best_y = {}

    ###############################################################

    model_mse_all = []
    model_mse_std_all = []
    for iNpoints in n_points_to_fit:
        model_mse_runs = []
        print("Analyse mit %d-Punkten ...." % (iNpoints))
        for i in range(n_runs):
            x_p, y_p = getInitPoints(start_points, iNpoints, min_x, max_x, min_y, max_y)
            z_p = f1.getValue(x_p, y_p)

            best_model, best_score, bin_center, gamma = getBestCovarianceModel(x_p, y_p, z_p, False)

            krig = calculateKrigingModel(krig_type, best_model, x_p, y_p, z_p)

            krige_field, krige_var = krig((x, y), return_var=True, only_mean=False)

            curr_mse = getMSE(z, krige_field)

            model_mse_runs.append(curr_mse)

            if curr_mse < best_mse_all:
                best_mse_all = curr_mse
                best_krige = krig
                best_x = x_p
                best_y = y_p
                best_cov_model = best_model
                best_bin_center = bin_center
                best_gamma = gamma

        mse_mean = np.mean(model_mse_runs)
        mse_std = np.std(model_mse_runs)



        model_mse_all.append(mse_mean)
        model_mse_std_all.append(mse_std)

        print("MSE mit %d-Punkten: %.5f +/- %.5f" % (iNpoints, mse_mean, mse_std))

    model_mse.append(model_mse_all)
    model_mse_std.append(model_mse_std_all)

    ###############################################################

    model_mse_all = []
    model_mse_std_all = []
    for iNpoints in n_points_to_fit:
        model_mse_runs = []
        print("Analyse mit %d-Punkten (zusätzliche Punkte)...." % (iNpoints))
        for i in range(n_runs):

            nInitPoints = int(np.floor(iNpoints*init_faction))

            x_p, y_p = getInitPoints(start_points, iNpoints, min_x, max_x, min_y, max_y)
            z_p = f1.getValue(x_p, y_p)

            best_model, best_score, bin_center, gamma = getBestCovarianceModel(x_p, y_p, z_p, False)

            krig = calculateKrigingModel(krig_type, best_model, x_p, y_p, z_p)

            krige_field, krige_var = krig((x, y), return_var=True, only_mean=False)

            for iAddPoint in range(iNpoints-nInitPoints):
                x_new, y_new = getPointWithMaxValue(krig, min_x, max_x, min_y, max_y)
                z_new = f1.getValue(x_new, y_new)
                x_p = np.append(x_p, x_new)
                y_p = np.append(y_p, y_new)
                z_p = np.append(z_p, z_new)

                best_model, best_score, bin_center, gamma = getBestCovarianceModel(x_p, y_p, z_p, False)

                krig = calculateKrigingModel(krig_type, best_model, x_p, y_p, z_p)
                krige_field, krige_var = krig((x, y), return_var=True, only_mean=False)

            curr_mse = getMSE(z, krige_field)

            model_mse_runs.append(curr_mse)

            if curr_mse < best_mse_all:
                best_mse_all = curr_mse
                best_krige = krig
                best_x = x_p
                best_y = y_p
                best_cov_model = best_model
                best_bin_center = bin_center
                best_gamma = gamma

        mse_mean = np.mean(model_mse_runs)
        mse_std = np.std(model_mse_runs)

        model_mse_all.append(mse_mean)
        model_mse_std_all.append(mse_std)

        print("MSE mit %d-Punkten (zusätzliche Punkte): %.5f +/- %.5f" % (iNpoints, mse_mean, mse_std))

    model_mse.append(model_mse_all)
    model_mse_std.append(model_mse_std_all)


    ########################################################################################


    ax1.errorbar(n_points_to_fit, model_mse[0], yerr= model_mse_std[0], fmt='-o', color="blue",ecolor = 'blue', label='alle')
    ax1.set_title('MSE vs #-Stützpunkte')
    ax1.set_xlabel("Anzahl Datenpunkte")
    ax1.set_ylabel("MSE")
    ax1.errorbar(n_points_to_fit, model_mse[1], yerr=model_mse_std[1], fmt='-o', color="green", ecolor='green', label='init %.2f + einzeln'%init_faction)
    ax1.legend(loc="upper right")

    plotKrigModel(best_krige, min_x, max_x, min_y, max_y, best_x, best_y, fig, ax2, ax3)

    plotCovarianceModel(bin_center, gamma, best_cov_model, ax4)

    fig.set_size_inches(30, 5)

    return model_mse_all, model_mse_std_all, best_krige, best_cov_model, best_bin_center, best_gamma

    best_krige_field, best_krige_var = best_krige((x, y), return_var=True, only_mean=False)
    
    
def analyseMSEvsN(f1, n_points_to_fit_min, n_points_to_fit_max, start_points_distribution, covariance_function, krig_types, n_runs, n_points_to_analyse):

    min_x, max_x, min_y, max_y = f1.getRange() 

    n_points_to_plot = 100
    x, y = np.meshgrid(np.linspace(min_x, max_x, n_points_to_plot), np.linspace(min_y, max_y, n_points_to_plot))
    z = f1.getValue(x, y)

    model_mse = []
    
    all_parameters = []
    
    analysed_values = []
    
    for iPointsFunction in start_points_distribution:
        for iCovFunction in covariance_function:
            for iKrigType in krig_types:
                for iNpoints in range(n_points_to_fit_min,n_points_to_fit_max):
                    all_parameters.append((iPointsFunction, iCovFunction,  iKrigType, iNpoints))
    n_parameters = len(all_parameters)
    if n_points_to_analyse < n_parameters:
        rd.shuffle(all_parameters)
        all_parameters = all_parameters[:n_points_to_analyse]
        n_parameters =  n_points_to_analyse
    iRun = 1                
    for iParamSet in all_parameters:
        model_mse_runs = []
        
        iNpoints = iParamSet[3]
        start_points = iParamSet[0]
        krig_type = iParamSet[2]
        cov_function = iParamSet[1]
        
        
        
        
        for i in range(n_runs):
            try:
                x_p, y_p = getInitPoints(start_points, iNpoints, min_x, max_x, min_y, max_y)
                z_p = f1.getValue(x_p, y_p)

                best_model, best_score, bin_center, gamma =  calculateCovarianceModel(x_p, y_p, z_p, cov_function)

                krig = calculateKrigingModel(krig_type, best_model, x_p, y_p, z_p)

                krige_field, krige_var = krig((x, y), return_var=True, only_mean=False)

                curr_mse = getMSE(z, krige_field)

                model_mse_runs.append(curr_mse)
            except Exception as e:
                print("Parameter-Set %d of %d: calculate(%s, %s, %s, %d) failed" % (iRun,n_parameters ,krig_type, cov_function, start_points, iNpoints))

        mse_mean = np.mean(model_mse_runs)
        analysed_values.append(( krig_type, cov_function, start_points, iNpoints, mse_mean))
        print("Parameter-Set %d of %d: calculate(%s, %s, %s, %d) with MSE =  %.5f" % (iRun,n_parameters ,krig_type, cov_function, start_points, iNpoints, mse_mean))
        iRun += 1
   

    

            
    return analysed_values
            
 
    
def plotCovarianceModel(bin_center, gamma, fit_model,ax):

    ax.scatter(bin_center, gamma, color="k", label="data")
    max_distance_variogram =  max(bin_center)*1.5
    fit_model.plot(x_max=max_distance_variogram, ax=ax)
    ax.title.set_text('Variogram')


def getSEmapOfKrigModel(krig_model, fkt, min_x, max_x, min_y, max_y, x_p, y_p, fig, ax1_in):
    n_points_to_plot = 100
    x, y = np.meshgrid(np.linspace(min_x, max_x, n_points_to_plot), np.linspace(min_y, max_y, n_points_to_plot))

    krige_field, krige_var = krig_model((x, y), return_var=True, only_mean=False)
    krige_field_reshaped = np.reshape(krige_field, (n_points_to_plot, n_points_to_plot))


    #calculate true values
    z = fkt.getValue(x,y)

    se_map = (z - krige_field_reshaped)**2

    mse = getMSE(z, krige_field)
    # plot se function
    z_min = np.min(se_map)
    z_max = np.max(se_map)



    levels = np.linspace(z_min, z_max, n_points_to_plot + 1)
    contourf_ = ax1_in.contourf(x, y, se_map, levels, vmin=z_min, vmax=z_max, cmap='viridis')
    ax1_in.title.set_text('Squared error map (exact <> predicted) with MSE of %.5f' %mse)
    fig.colorbar(contourf_, ax=ax1_in)
    ax1_in.scatter(x_p, y_p, color="red")
    return se_map, mse



def getBestCovarianceModel(x_p, y_p, z_p, textoutput = False, models = {
        "Gaussian": gs.Gaussian,
        "Exponential": gs.Exponential,
        "Matern": gs.Matern,
        "Stable": gs.Stable,
        "Rational": gs.Rational,
        "Circular": gs.Circular,
        "Spherical": gs.Spherical,
        "SuperSpherical": gs.SuperSpherical
    }):


    bin_center, gamma = gs.vario_estimate((x_p, y_p), z_p)
    bin_center = bin_center[gamma > 0]
    gamma = gamma[gamma > 0]

    best_model = {}
    best_score = -1
    best_idx = "not_found"

    # fit all models to the estimated variogram
    for model in models:
        fit_model = models[model](dim=2)
        try:
            para, pcov, r2 = fit_model.fit_variogram(bin_center, gamma, return_r2=True, nugget=False)
            if r2 > best_score:
                best_score = r2
                best_model = fit_model

                best_idx = model
        except:
            skip_it = 1



    if textoutput == True:
        print("Bestes Kovarianzmodel: %s mit r2 = %.5f"%(best_idx, best_score))
    return best_model, best_score, bin_center, gamma

def calculateCovarianceModel(x_p, y_p, z_p, model):
    cur_model = {"Gaussian": gs.Gaussian}
    
    if model.lower() == "gaussian":
        cur_model = {"Gaussian": gs.Gaussian}
        
    elif model.lower() == "exponential":
        cur_model = {"Exponential": gs.Exponential}
        
    elif model.lower() == "matern":
        cur_model = {"Matern": gs.Matern}
        
    elif model.lower() == "stable":
        cur_model = {"Stable": gs.Stable}
        
    elif model.lower() == "rational":
        cur_model = {"Rational": gs.Rational}
        
    elif model.lower() == "circular":
        cur_model = {"Circular": gs.Circular}
        
    elif model.lower() == "spherical":
        cur_model = {"Spherical": gs.Spherical}
        
    elif model.lower() == "superspherical":
        cur_model = {"SuperSpherical": gs.SuperSpherical}
        
    elif model.lower() == "selectbest":
        cur_model = {
        "Gaussian": gs.Gaussian,
        "Exponential": gs.Exponential,
        "Matern": gs.Matern,
        "Stable": gs.Stable,
        "Rational": gs.Rational,
        "Circular": gs.Circular,
        "Spherical": gs.Spherical,
        "SuperSpherical": gs.SuperSpherical
        }        
    else:
        print("Überprüfe covariance model: %s. Gaussian model wird anstatt verwendet" % (model) )

    return getBestCovarianceModel(x_p, y_p, z_p, False, cur_model)
    

def getPointWithMaxValue(krig_model, min_x, max_x, min_y, max_y):
    n_points_to_plot = 100
    x, y = np.meshgrid(np.linspace(min_x, max_x, n_points_to_plot), np.linspace(min_y, max_y, n_points_to_plot))
    krige_field, krige_var = krig_model((x, y), return_var=True, only_mean=False)
    x_res = np.reshape(x, len(krige_var))
    y_res = np.reshape(y, len(krige_var))
    max_idx = np.argmax(krige_var)
    return x_res[max_idx], y_res[max_idx]

def addOnePoint(x_new, y_new, x_current, y_current):
    x_extended = np.append(x_current, x_new)
    y_extended = np.append(y_current, y_new)
    return x_extended, y_extended

def getMSE(z,field):
    n_len = len(field)
    z_arr1d = np.reshape(z, n_len)
    return ((field-z_arr1d)**2).mean()

def addRandomPoints(n, x_p_in, y_p_in):
    n_curr = len(x_p_in)
    if n_curr < n:
        min_x_curr = min(x_p_in)
        max_x_curr = max(x_p_in)
        min_y_curr = min(y_p_in)
        max_y_curr = max(y_p_in)
        (x_p_curr_add, y_p_curr_add) = getRandomPoints(n-n_curr, min_x_curr, max_x_curr, min_y_curr, max_y_curr)
        x_p_curr_add = np.concatenate((x_p_curr_add, x_p_in))
        y_p_curr_add = np.concatenate((y_p_curr_add, y_p_in))
        return x_p_curr_add, y_p_curr_add
    else:
        return x_p_in, y_p_in

#+vier punkte nahe der ecken
def addEdgePoints(x_values_in, y_values_in, min_x, max_x, min_y,max_y):
    n_per_line = int(round(np.floor(np.sqrt(len(x_values_in)))))
    x_shift = 0.2 * (max_x - min_x) / (n_per_line)
    y_shift = 0.2 * (max_y - min_y) / (n_per_line)
    x_values = np.concatenate((x_values_in, np.asarray([min_x+x_shift ,min_x+x_shift,max_x-x_shift,max_x-x_shift])))
    y_values = np.concatenate((y_values_in, np.asarray([min_y+y_shift,max_y-y_shift,min_y+y_shift,max_y-y_shift])))
    return x_values, y_values

#zufällige verteilung
def getRandomPoints(n, min_x,max_x, min_y,max_y):
    return  np.random.uniform(min_x, max_x, int(n)), np.random.uniform(min_y, max_y, int(n))

#im gitter angeordnet mit randpunkten
def getGridPoints(n, min_x,max_x, min_y,max_y):
    n_per_line = round(np.floor(np.sqrt(n)))
    x_1 = np.linspace(min_x,max_x,n_per_line)
    y_1 = np.linspace(min_y, max_y, n_per_line)
    x_values = []
    y_values = []
    for x_curr in x_1:
        for y_curr in y_1:
            x_values.append(x_curr)
            y_values.append(y_curr)
    return addRandomPoints(n, x_values, y_values)

#im gitter angeordnet, ohne randpunkte
def getCenteredGridPoints(n, min_x,max_x, min_y,max_y):
    n_per_line = round(np.floor(np.sqrt(n)))
    x_center = 0.5*(max_x-min_x) / (n_per_line)
    y_center = 0.5*(max_y - min_y) / (n_per_line)
    x_1 = np.linspace(min_x+x_center ,max_x-x_center ,n_per_line)
    y_1 = np.linspace(min_y+y_center, max_y-y_center, n_per_line)
    x_values = []
    y_values = []
    for x_curr in x_1:
        for y_curr in y_1:
            x_values.append(x_curr)
            y_values.append(y_curr)
    return addRandomPoints(n, x_values, y_values)

#in gitter aufgeteilt und im quadranten eine zufallszahl gewählt
def getCenteredRandomGridPoints(n, min_x,max_x, min_y,max_y):
    x_p_curr, y_p_curr = getCenteredGridPoints(n, min_x, max_x, min_y, max_y)
    n_per_line = round(np.floor(np.sqrt(n)))
    x_center = 0.5 * (max_x - min_x) / (n_per_line)
    y_center = 0.5 * (max_y - min_y) / (n_per_line)
    x_values = []
    y_values = []
    for i in range(len(x_p_curr)):
        r_x = (np.random.uniform(0, 1)-0.5)*x_center
        r_y = (np.random.uniform(0, 1)-0.5)*y_center
        x_values.append(x_p_curr[i]+ r_x)
        y_values.append(y_p_curr[i] + r_y)
    return addRandomPoints(n, x_values, y_values)

#kombination auf random-grid und random punkten
def getCenteredRandomGridCombinedPoints(n, min_x,max_x, min_y,max_y):
    fraction_grid = 0.75
    x_values, y_values = getCenteredRandomGridPoints(np.round(n*fraction_grid), min_x, max_x, min_y, max_y)
    return addRandomPoints(n, x_values, y_values)

def getCenteredRandomGridCombinedPointsWithEdges(n, min_x,max_x, min_y,max_y):
    fraction_grid = 0.75
    x_values, y_values = getCenteredRandomGridPoints(np.round(n*fraction_grid), min_x, max_x, min_y, max_y)
    if np.round(n*(1-fraction_grid)) > 3:
        x_values, y_values = addEdgePoints(x_values, y_values, min_x, max_x, min_y,max_y)
    return addRandomPoints(n, x_values, y_values)

#distribution_type "RandomPoints" (bzw. "else-case),  "GridPoints" / "CenteredGridPoints" /"CenteredRandomGridPoints" / "CenteredRandomGridCombinedPoints" / "CenteredRandomGridCombinedPointsWithEdges"
def getInitPoints(selection_type, n, min_x,max_x, min_y,max_y):

    if selection_type == "GridPoints":
        return getGridPoints(n, min_x,max_x, min_y,max_y)
    elif selection_type == "CenteredGridPoints":
        return getCenteredGridPoints(n, min_x,max_x, min_y,max_y)
    elif selection_type == "CenteredRandomGridPoints":
        return getCenteredRandomGridPoints(n, min_x,max_x, min_y,max_y)
    elif selection_type == "CenteredRandomGridCombinedPoints":
        return getCenteredRandomGridCombinedPoints(n, min_x,max_x, min_y,max_y)
    elif selection_type == "CenteredRandomGridCombinedPointsWithEdges":
        return getCenteredRandomGridCombinedPointsWithEdges(n, min_x,max_x, min_y,max_y)
    elif selection_type == "LHS":
        return getLHSPoints(n, min_x,max_x, min_y,max_y)
    elif selection_type == "MC":
        return getLHSPoints(n, min_x,max_x, min_y,max_y)
    else:
        return getRandomPoints(n, min_x,max_x, min_y,max_y)

    return []

#latice hypercube sampling
def getLHSPoints(n, min_x,max_x, min_y,max_y):
    l = lhs.sample(2,n) # Latin Hypercube Sampling of two variables, and n samples each.
    x_p = np.squeeze(np.asarray(min_x + l[0]*(max_x-min_x)))
    y_p = np.squeeze(np.asarray(min_y + l[1]*(max_y-min_y)))
    return x_p, y_p

#monte carlo sampling
def getMCPoints(n, min_x,max_x, min_y,max_y):
    k = lhs.createRandomStandardUniformMatrix(2,n) # Monte Carlo Sampling
    x_p = np.squeeze(np.asarray(min_x + k[0]*(max_x-min_x)))
    y_p = np.squeeze(np.asarray(min_y + k[1]*(max_y-min_y)))
    return x_p, y_p

def plotFunction2D(f, ax):

    min_x, max_x, min_y, max_y = f.getRange()

    n_points_to_plot = 100
    x, y = np.meshgrid(np.linspace(min_x, max_x, n_points_to_plot), np.linspace(min_y, max_y, n_points_to_plot))
    z = f.getValue(x, y)
    z_min_loc = np.min(z)
    z_max_loc = np.max(z)

    levels_loc = np.linspace(z_min_loc, z_max_loc, n_points_to_plot + 1)

    # plot true function
    contourf_ = ax.contourf(x, y, z, levels_loc, vmin=z_min_loc, vmax=z_max_loc, cmap='viridis')
    plt.colorbar(contourf_, ax=ax)


########################################################################
########## die funktionen sind als klassen definiert ###################
############ Werteberech ist anpassbar #################################
############  getValue() ist anpassbar #################################
########################################################################

class f1:
    def __init__(self):
        #definiere Wertebereich der Funktion
        #kann und sollte angepasst werden
        self.min_x = 0
        self.max_x = np.pi
        self.min_y = 0
        self.max_y = np.pi

    def getRange(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def getValue(self, x, y):
        #definiere funktion
        #kann und sollte angepasst werden
        p1 = np.poly1d([0.8, -2.7, 1, 1])
        return p1(x) * (np.tanh(y))

class f2:
    def __init__(self):
        #definiere Wertebereich der Funktion
        #kann uns sollte angepasst werden
        self.min_x = 0
        self.max_x = np.pi
        self.min_y = 0
        self.max_y = np.pi

    def getRange(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def getValue(self, x, y):
        #definiere funktion
        #kann und sollte angepasst werden
        p1 = np.poly1d([0.3461, 0.2418, - 0.3908, -10])
        p2 = np.poly1d([-0.2142, - 0.3593, 0.2363, - 0.2625])
        return p1(x)*p2(y)*np.sin(x)*np.cos(x)/20

class f3:
    def __init__(self):
        #definiere Wertebereich der Funktion
        #kann uns sollte angepasst werden
        self.min_x = 0
        self.max_x = np.pi
        self.min_y = 0
        self.max_y = np.pi

    def getRange(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def getValue(self, x, y):
        #definiere funktion
        #kann und sollte angepasst werden
        p1 = np.poly1d([0.3461, 0.2418, - 0.3908, -10])
        p2 = np.poly1d([-0.2142, - 0.3593, 0.2363, - 0.2625])
        return p1(x)*p2(y)*np.sin(x)*np.cos(y)/20

class f_herbie_1:
    def __init__(self):
        #definiere Wertebereich der Funktion
        #kann und sollte angepasst werden
        self.min_x = 0
        self.max_x = 2
        self.min_y = 0
        self.max_y = 2

    def getRange(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def getValue(self, x, y):
        #definiere funktion
        #kann und sollte angepasst werden
        return (np.exp(-(x - 1) ** 2) + np.exp(-0.8 * (x + 1) ** 2)) * (np.exp(-(y - 1) ** 2) + np.exp(-0.8 * (y + 1) ** 2))

class f_herbie_2:
    def __init__(self):
        #definiere Wertebereich der Funktion
        #kann und sollte angepasst werden
        self.min_x = -1
        self.max_x = 1
        self.min_y = -1
        self.max_y = 1

    def getRange(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def getValue(self, x, y):
        #definiere funktion
        #kann und sollte angepasst werden
        return (np.exp(-(x - 1) ** 2) + np.exp(-0.8 * (x + 1) ** 2)) * (np.exp(-(y - 1) ** 2) + np.exp(-0.8 * (y + 1) ** 2))

class f_herbie_3:
    def __init__(self):
        #definiere Wertebereich der Funktion
        #kann und sollte angepasst werden
        self.min_x = -0.5
        self.max_x = 1.5
        self.min_y = -0.5
        self.max_y = 1.5

    def getRange(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def getValue(self, x, y):
        #definiere funktion
        #kann und sollte angepasst werden
        return (np.exp(-(x - 1) ** 2) + np.exp(-0.8 * (x + 1) ** 2)) * (np.exp(-(y - 1) ** 2) + np.exp(-0.8 * (y + 1) ** 2))

class f_herbie_4:
    def __init__(self):
        #definiere Wertebereich der Funktion
        #kann und sollte angepasst werden
        self.min_x = 0
        self.max_x = 1.5
        self.min_y = 0
        self.max_y = 1.5

    def getRange(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def getValue(self, x, y):
        #definiere funktion
        #kann und sollte angepasst werden
        return (np.exp(-(x - 1) ** 2) + np.exp(-0.8 * (x + 1) ** 2)) * (np.exp(-(y - 1) ** 2) + np.exp(-0.8 * (y + 1) ** 2))

class f_herbie_complex:
    def __init__(self):
        #definiere Wertebereich der Funktion
        #kann und sollte angepasst werden
        self.min_x = -2
        self.max_x = 2
        self.min_y = -2
        self.max_y = 2

    def getRange(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def getValue(self, x, y):
        #definiere funktion
        #kann und sollte angepasst werden
        return (np.exp(-(x - 1) ** 2) + np.exp(-0.8 * (x + 1) ** 2)) * (np.exp(-(y - 1) ** 2) + np.exp(-0.8 * (y + 1) ** 2))

class f_rosenbrock:
    def __init__(self):
        #definiere Wertebereich der Funktion
        #kann und sollte angepasst werden
        self.min_x = -2
        self.max_x = 2
        self.min_y = -2
        self.max_y = 2

    def getRange(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def getValue(self, x, y):
        #definiere funktion
        #kann und sollte angepasst werden
        a = 1
        b = 10
        return ((a-x)**2+b*(y-x**2)**2)/3000

class f_zum_versagen_von_kriging:
    def __init__(self):
        #definiere Wertebereich der Funktion
        #kann und sollte angepasst werden
        self.min_x = -1
        self.max_x = 1
        self.min_y = -1
        self.max_y = 1

    def getRange(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def getValue(self, x, y):
        #definiere funktion
        #kann und sollte angepasst werden
        f = 0
        for i in range(5):
            f = f + (i*np.cos((i+1)*x+i))*(i*np.cos((i+1)*y+i))*(x-y)**2/ 18
        R = np.sqrt(x**2+y**2)
        freq = 1
        f = f + 2*np.sin(np.pi*R*freq)/(np.pi*R*freq)
        return f

class f_shubert_1_not_for_hand_ons:
    def __init__(self):
        #definiere Wertebereich der Funktion
        #kann und sollte angepasst werden
        self.min_x = 0
        self.max_x = 5
        self.min_y = -3.2
        self.max_y = 1.8

    def getRange(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def getValue(self, x, y):
        #definiere funktion
        #kann und sollte angepasst werden
        f = 0
        for i in range(5):
            f = f + (i*np.cos((i+1)*x+i))*(i*np.cos((i+1)*y+i))/ 18
        return f

class f_shubert_2_not_for_hand_ons:
    def __init__(self):
        #definiere Wertebereich der Funktion
        #kann und sollte angepasst werden
        self.min_x = 0
        self.max_x = 5
        self.min_y = -3.2
        self.max_y = 1.8

    def getRange(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def getValue(self, x, y):
        #definiere funktion
        #kann und sollte angepasst werden
        f = 0
        r_value = 0.5
        r = 0
        if hasattr(x, "__len__"):
            r = np.random.uniform(-r_value, r_value, int(x.size))
            if len(x)>0 and hasattr(x[0], "__len__"):
                print("2d-array")
                r = np.reshape(r, (len(x[0]), len(x[1])))
            else:
                r = np.reshape(r, len(x))
        elif type(x) is float or type(x) is int or type(x):
            r = np.random.uniform(-r_value, r_value, int(1))
            print("scalar")



        for i in range(5):
            f = f + (i*np.cos((i+1)*x+i))*(i*np.cos((i+1)*y+i))/ 18
        return f + r

class f_parabolid:
    def __init__(self):
        #definiere Wertebereich der Funktion
        #kann und sollte angepasst werden
        self.min_x = -2
        self.max_x = 2
        self.min_y = -2
        self.max_y = 2

    def getRange(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def getValue(self, x, y):
        #definiere funktion
        #kann und sollte angepasst werden
        return x**2+y**2

class f_hyperbolic_parabolid:
    def __init__(self):
        #definiere Wertebereich der Funktion
        #kann und sollte angepasst werden
        self.min_x = -2
        self.max_x = 2
        self.min_y = -2
        self.max_y = 2

    def getRange(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def getValue(self, x, y):
        #definiere funktion
        #kann und sollte angepasst werden
        return x**2-y**2

class f_eggholder:
    def __init__(self):
        #definiere Wertebereich der Funktion
        #kann und sollte angepasst werden
        self.min_x = 0
        self.max_x = 250
        self.min_y = -512
        self.max_y = -262

    def getRange(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def getValue(self, x, y):
        #definiere funktion
        #kann und sollte angepasst werden
        return (-(y+47)*np.sin(np.sqrt(np.abs(y+x/2.+47)))-x*np.sin(np.sqrt(np.abs(x-(y+47)))))/250

class f_six_hump_camel:
    def __init__(self):
        #definiere Wertebereich der Funktion
        #kann und sollte angepasst werden
        self.min_x = -1
        self.max_x = 1.5
        self.min_y = -1.25
        self.max_y = 1.25

    def getRange(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def getValue(self, x, y):
        #definiere funktion
        #kann und sollte angepasst werden
        return (4.-2.1*x**2+x**4/3)*x**2+x*y+(-4 + 4*y**2)*y**2

class f_three_hump_camel:
    def __init__(self):
        #definiere Wertebereich der Funktion
        #kann und sollte angepasst werden
        self.min_x = -1
        self.max_x = 2
        self.min_y = -1.5
        self.max_y = 1.5

    def getRange(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def getValue(self, x, y):
        #definiere funktion
        #kann und sollte angepasst werden
        return (2*x**2 - 1.05*x**4+x**6/6+x*y+y**2)

class f_branin:
    def __init__(self):
        #definiere Wertebereich der Funktion
        #kann und sollte angepasst werden
        self.min_x = 0
        self.max_x = 10
        self.min_y = 5
        self.max_y = 15

    def getRange(self):
        return self.min_x, self.max_x, self.min_y, self.max_y

    def getValue(self, x, y):
        #definiere funktion
        #kann und sollte angepasst werden
        a = 1.
        b = 5.1/(4*np.pi**2)
        c = 5/np.pi
        r = 6
        s = 10
        t = 1/(8*np.pi)
        return (a*(y-b*x**2+c*x-r)**2+s*(1-t)*np.cos(x)+s)/210

if __name__ == '__main__':
    f_test = f_branin()
    fig, ax = plt.subplots()

    plotFunction2D(f_test,ax)

    plt.show()

"""

if __name__ == '__main__':
    #wähle die funktion aus
    f_current = f_hyperbolic_parabolid()

    #extrahiere wertebereich
    min_x, max_x, min_y, max_y = f_current.getRange()

    fig, ax = plt.subplots()
    plotFunction2D(f_current, ax)

    f_current.getValue(0.1, 0.2)

    plt.show()



    #definiere parameter
    n_known_datapoints = 50
    n_points_to_plot = 100

    #distribution_type
    # / "RandomPoints" (bzw. "else-case),
    # / "GridPoints"
    # / "CenteredGridPoints"
    # / "CenteredRandomGridPoints"
    # / "CenteredRandomGridCombinedPoints"
    # / "CenteredRandomGridCombinedPointsWithEdges"
    x_p,y_p = getInitPoints("CenteredRandomGridCombinedPoints", n_known_datapoints, min_x,max_x, min_y,max_y)

    z_p = f_current.getValue(x_p,y_p)



    #bins = np.linspace(0,max_distance_variogram,round(n_known_datapoints/3))
    bin_center, gamma = gs.vario_estimate((x_p,y_p), z_p)
    bin_center = bin_center[gamma > 0]
    gamma = gamma[gamma > 0]

    models = {
        "Gaussian": gs.Gaussian,
        "Exponential": gs.Exponential,
        "Matern": gs.Matern,
        "Stable": gs.Stable,
        "Rational": gs.Rational,
        "Circular": gs.Circular,
        "Spherical": gs.Spherical,
        "SuperSpherical": gs.SuperSpherical,
        #"JBessel": gs.JBessel, #keine Varianzberechnung möglich !!
    }


    # plot the estimated variogram
    plt.scatter(bin_center, gamma, color="k", label="data")
    ax = plt.gca()

    # fit all models to the estimated variogram
    scores = {}
    max_distance_variogram = 0.75*max(max_x-min_x, max_y-min_y)
    for model in models:
        fit_model = models[model](dim=2)
        para, pcov, r2 = fit_model.fit_variogram(bin_center, gamma, return_r2=True, nugget=False)
        fit_model.plot(x_max=max_distance_variogram, ax=ax)
        scores[model] = r2

    ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    print("RANKING by Pseudo-r2 score")
    best_model = {}
    best_score = -1
    for i, (model, score) in enumerate(ranking, 1):
        if score > best_score:
            best_score = score
            best_model = models[model](dim=2)
        print(f"{i:>6}. {model:>15}: {score:.5}")


    x,y = np.meshgrid(np.linspace(min_x,max_x,n_points_to_plot),np.linspace(min_y,max_y,n_points_to_plot))
    z = f_current.getValue(x, y)
    z_arr1d = np.reshape(z, n_points_to_plot**2)

    #
    #




    krig = calculateKrigingModel("universal_linear", best_model, x_p,y_p, z_p)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plotKrigModel(krig, min_x, max_x, min_y, max_y, x_p, y_p, fig, ax1, ax2)

    plt.show()


    krige_field, krige_var = krig((x,y), return_var=True, only_mean=False)
    x_max_se, y_max_se = getPointWithMaxValue(x,y,krige_var)

    krige_field_reshaped = np.reshape(krige_field, (n_points_to_plot, n_points_to_plot))


    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Wahre Funktion vs. Kriging')

    z_min = np.min(np.concatenate((z_arr1d, krige_field)))
    z_max = np.max(np.concatenate((z_arr1d, krige_field)))


    levels = np.linspace(z_min, z_max, n_points_to_plot+1)

    #plot true function
    contourf_ = ax1.contourf(x,y,z,levels,vmin=z_min,vmax=z_max,cmap='viridis')
    ax1.scatter(x_p,y_p, color = "red")
    ax1.title.set_text('Funktion')

    #plot predicted function
    contourf_ = ax2.contourf(x,y,krige_field_reshaped,levels,vmin=z_min,vmax=z_max,cmap='viridis')
    #plt.colorbar(contourf_, ax = 2)
    ax2.scatter(x_p,y_p, color = "red")
    ax2.title.set_text('Kriging')
    fig.colorbar(contourf_)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    z_se = (z-krige_field_reshaped)**2
    z_std_arr1d = np.sqrt(krige_var)
    z_std = np.reshape(z_std_arr1d, (n_points_to_plot, n_points_to_plot))

    z_se_arr1d = np.reshape(z_se, n_points_to_plot**2)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('SE vs. Kriging Std.Abweichung')

    z_min = np.min(z_se_arr1d)
    z_max = np.max(z_se_arr1d)
    levels = np.linspace(z_min, z_max, n_points_to_plot+1)
    contourf_ = ax1.contourf(x,y,z_se,levels,vmin=z_min,vmax=z_max,cmap='viridis')
    ax1.scatter(x_p,y_p, color = "red")
    ax1.title.set_text('SE: %.3f'%getMSE(z, krige_field))
    plt.colorbar(contourf_, ax=ax1)

    z_min = np.min(z_std_arr1d)
    z_max = np.max(z_std_arr1d)
    levels = np.linspace(z_min, z_max, n_points_to_plot+1)
    contourf_ = ax2.contourf(x,y,z_std,levels,vmin=z_min,vmax=z_max,cmap='viridis')
    ax2.scatter(x_p,y_p, color = "red")
    ax2.title.set_text('Kriging Std.Abweichung-> mean(): %.3f'%np.mean(z_std_arr1d))
    plt.colorbar(contourf_, ax=ax2)
    ax2.scatter(x_max_se, y_max_se, color = "black")


    #plt.show()



    #############################################################################
    ################# HAND-ON: Variogrammen erstellen und verstehen #############
    #############################################################################


    ################# HAND-ON: Varianzaufgabe 1 #############

    #Erstelle deine eigne funktion, oder verwende eine der bereits definierten
    class f_deine_eigene_funktion:
        def __init__(self):
            #definiere Wertebereich der Funktion
            self.min_x = 0
            self.max_x = 1
            self.min_y = 0
            self.max_y = 1

        def getRange(self):
            return self.min_x, self.max_x, self.min_y, self.max_y

        def getValue(self, x, y):
            #definiere funktion
            return x + y

    #funktion zum anschauen der funktion
    #plotFunction2D(f_deine_eigene_funktion)

    #wähle die funktion aus
    f_bekannte_funktion = f_deine_eigene_funktion()

    #extrahiere wertebereich
    min_x, max_x, min_y, max_y = f_bekannte_funktion.getRange()


    #fitte variogram (bi dieser Aufgabe gerne mehr Punkte verwenden, es geht ums Verständnis der Variogramme
    #startpunkte
    n_known_datapoints = 100

    #distribution_type
    # / "RandomPoints" (bzw. "else-case),
    # / "GridPoints"
    # / "CenteredGridPoints"
    # / "CenteredRandomGridPoints"
    # / "CenteredRandomGridCombinedPoints"
    # / "CenteredRandomGridCombinedPointsWithEdges"
    x_p,y_p = getInitPoints("CenteredRandomGridCombinedPoints", n_known_datapoints, min_x,max_x, min_y,max_y)
    z_p = f_bekannte_funktion.getValue(x_p,y_p)


    #berechne experimentelles variogram

    bin_center, gamma = gs.vario_estimate((x_p, y_p), z_p)
    #enferne fehlende distanzen
    bin_center = bin_center[gamma > 0]
    gamma = gamma[gamma > 0]

    #schaue das ergebnis an
    #welche fit-funktion würde sich hier anbieten ?
    fig, ax = plt.subplots()
    # plot the estimated variogram
    ax.scatter(bin_center, gamma, color="k", label="data")
    plt.show()

    #wähle deine favoriten aus
    models = {
        "Gaussian": gs.Gaussian,
        #"Exponential": gs.Exponential,
        #"Matern": gs.Matern,
        #"Stable": gs.Stable,
        #"Rational": gs.Rational,
        #"Circular": gs.Circular,
        #"Spherical": gs.Spherical,
        #"SuperSpherical": gs.SuperSpherical,
        #"JBessel": gs.JBessel, #keine Varianzberechnung möglich !!
    }

    #betrachte nun die gefittetes covarianz-modell
    # plot the estimated variogram
    plt.scatter(bin_center, gamma, color="k", label="data")
    ax = plt.gca()

    # fit all models to the estimated variogram
    scores = {}
    max_distance_variogram = 0.75*max(max_x-min_x, max_y-min_y)
    for model in models:
        fit_model = models[model](dim=2)
        para, pcov, r2 = fit_model.fit_variogram(bin_center, gamma, return_r2=True, nugget=False)
        fit_model.plot(x_max=max_distance_variogram, ax=ax)
        scores[model] = r2

    ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    print("RANKING by Pseudo-r2 score")
    best_model = {}
    best_score = -1
    for i, (model, score) in enumerate(ranking, 1):
        if score > best_score:
            best_score = score
            best_model = models[model](dim=2)
        print(f"{i:>6}. {model:>15}: {score:.5}")


    #gehe zum vorherigen Punkt, reduziere die Anzahl der Stützpunkte und wiederhole die schritte
    #welche auswirkungen hat es auf variogram ?
    #ab welcher anzahl der Punkte würdest du kein zuverlässigen fit der kovarianz-funktion erwarten?




    ################# HAND-ON: Varianzaufgabe 2 #############
    f_var_2 = f_shubert_1_not_for_hand_ons()
    min_x, max_x, min_y, max_y = f_var_2.getRange()
    x_p,y_p = getInitPoints("CenteredRandomGridCombinedPoints", 3000, min_x,max_x, min_y,max_y)
    z_p = f_var_2.getValue(x_p,y_p)

    fig, (ax1, ax2) = plt.subplots(1,2)

    bins = np.linspace(0,3,40)
    bin_center, gamma = gs.vario_estimate((x_p,y_p), z_p, bin_edges=bins)
    bin_center = bin_center[gamma > 0]
    gamma = gamma[gamma > 0]

    models = {
        "JBessel": gs.JBessel,
    }


    # plot the estimated variogram
    ax2.scatter(bin_center, gamma, color="k", label="data")

    # fit all models to the estimated variogram
    for model in models:
        fit_model = models[model](dim=2)
        para, pcov, r2 = fit_model.fit_variogram(bin_center, gamma, return_r2=True, nugget=False)
        fit_model.plot(x_max=5, ax=ax2)

    plotFunction2D(f_var_2, ax1)

    f_var_3 = f_shubert_2_not_for_hand_ons()
    min_x, max_x, min_y, max_y = f_var_3.getRange()
    x_p,y_p = getInitPoints("CenteredRandomGridCombinedPoints", 3000, min_x,max_x, min_y,max_y)
    z_p = f_var_3.getValue(x_p,y_p)

    fig, (ax1, ax2) = plt.subplots(1,2)

    bins = np.linspace(0,3,40)
    bin_center, gamma = gs.vario_estimate((x_p,y_p), z_p, bin_edges=bins)
    bin_center = bin_center[gamma > 0]
    gamma = gamma[gamma > 0]

    models = {
        "JBessel": gs.JBessel,
    }


    # plot the estimated variogram
    ax2.scatter(bin_center, gamma, color="k", label="data")

    # fit all models to the estimated variogram
    for model in models:
        fit_model = models[model](dim=2)
        para, pcov, r2 = fit_model.fit_variogram(bin_center, gamma, return_r2=True, nugget=True)
        fit_model.plot(x_max=5, ax=ax2)

    plotFunction2D(f_var_3, ax1)

    #best_model, best_score, bin_center, gamma = getBestCovarianceModel(x_p, y_p, z_p, False)
    #plotCovarianceModel(bin_center, gamma, best_model, ax2)

    #plt.show()



    #############################################################################
    ################# HAND-ON: Kriging-Modelle erstellen und verstehen ##########
    #############################################################################

    ################# HAND-ON: Manuelle Auswahl der Punkte ##########

    #Erstelle deine eigne funktion, oder verwende eine der bereits definierten
    class f_deine_eigene_funktion_2:
        def __init__(self):
            #definiere Wertebereich der Funktion
            self.min_x = 0
            self.max_x = 1
            self.min_y = 0
            self.max_y = 1

        def getRange(self):
            return self.min_x, self.max_x, self.min_y, self.max_y

        def getValue(self, x, y):
            #definiere funktion
            return x + y



    #wähle die funktion aus
    f_bekannte_funktion_2 = f_deine_eigene_funktion_2()
    f_bekannte_funktion_2 = f_herbie_3()

    #funktion zum anschauen der funktion (objekt, nicht die klasse übergeben)
    #plotFunction2D(f_bekannte_funktion_2)

    #extrahiere wertebereich
    min_x, max_x, min_y, max_y = f_bekannte_funktion_2.getRange()

    #lege startpunkte manuell / visuell fest
    #entscheide dich für eine bestimmte anzahl der startpunkte (tipp: nicht weniger als 15, je nach fkt-komplexität)
    x_p = []
    y_p = []
    #beispiel zum hinzufügen von vier eckpunkten punkten (x,y)
    #für wertebereich 0-1
    #x_p , y_p = addOnePoint(0.0, 0.0, x_p , y_p)
    #x_p , y_p = addOnePoint(0.0, 1.0, x_p , y_p)
    #x_p , y_p = addOnePoint(1.0, 0.0, x_p , y_p)
    #x_p , y_p = addOnePoint(1.0, 1.0, x_p , y_p)

    #berechne die funktionswerte für startpunkte
    z_p = f_bekannte_funktion_2.getValue(x_p, y_p)

    #fitte variogram manuell (aus der variogram-aufgabe copy-pasten) oder automatisch mit der folgenden funktion

    best_model, best_score, bin_center, gamma = getBestCovarianceModel(x_p, y_p, z_p, True)

    #berechne kriging-model
    krig = calculateKrigingModel("universal_quadratic", best_model, x_p, y_p, z_p)

    #man kann das feld und sqrt(varianz) = std.abw anschauen
    fig, (ax1, ax2) = plt.subplots(1,2)
    plotKrigModel(krig, min_x, max_x, min_y, max_y, x_p, y_p, fig, ax1, ax2)

    fig, ax1 = plt.subplots()
    se_map, mse_manuell = getSEmapOfKrigModel(krig, f_bekannte_funktion_2, min_x, max_x, min_y, max_y, x_p, y_p, fig, ax1)

    print("MSE manuelle Datenpunkteauswahl: %.5f"%mse_manuell)

    ################# HAND-ON: Auswahl der Punkte mithilfe von max. erwarteten Modelfehler ##########
    ################# Die zu untersuchende Funktion bleibt aus manuellem Abschnitt         ##########


    #man muss mit bestimmten Anteil der Punkte starten (lass es mindestens 10 sein)
    #die restlichen Punkte (gleiche Anzahl wie bei der manuellen Selektion) werden einzeln hinzugefügt

    n_start_points = 25
    n_max_points = 30

    x_p,y_p = getInitPoints("CenteredRandomGridCombinedPointsWithEdges", n_start_points, min_x,max_x, min_y,max_y)
    z_p = f_bekannte_funktion_2.getValue(x_p, y_p)
    best_model, best_score, bin_center, gamma = getBestCovarianceModel(x_p, y_p, z_p, True)
    krig = calculateKrigingModel("universal_quadratic", best_model, x_p, y_p, z_p)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    plotKrigModel(krig, min_x, max_x, min_y, max_y, x_p, y_p, fig, ax1, ax2)
    plotCovarianceModel(bin_center, gamma, best_model, ax3)


    ########## begin der zu wiederholende teil, bis alle Punkte hingefügt sind ##########
    ## man kann zwar for-loop schreiben, wichtiger wäre allerdings zu festzuhalten, wie sich das model mit zusätzlichen
    ## punkten verhält

    for i in range(n_max_points - n_start_points):

        #hinfüge den punkt mit dem größten Modelfehler (manuell/visuell oder mit funktion)
        x_new, y_new = getPointWithMaxValue(krig, min_x, max_x, min_y, max_y)
        z_new = f_bekannte_funktion_2.getValue(x_new, y_new)
        x_p = np.append(x_p, x_new)
        y_p = np.append(y_p, y_new)
        z_p = np.append(z_p, z_new)

        best_model, best_score, bin_center, gamma = getBestCovarianceModel(x_p, y_p, z_p, True)

        #berechne kriging-model
        krig = calculateKrigingModel("universal_quadratic", best_model, x_p, y_p, z_p)

        #man kann das feld und sqrt(varianz) = std.abw anschauen
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        plotKrigModel(krig, min_x, max_x, min_y, max_y, x_p, y_p, fig, ax1, ax2)
        plotCovarianceModel(bin_center, gamma, best_model, ax3)


    ########## ende  ##########

    fig, ax1 = plt.subplots()
    se_map, mse_auto = getSEmapOfKrigModel(krig, f_bekannte_funktion_2, min_x, max_x, min_y, max_y, x_p, y_p, fig, ax1)

    print("MSE manuelle Datenpunkteauswahl: %.5f"%mse_manuell)
    print("MSE Datenpunkteauswahl nach größtem Modelfehler: %.5f"%mse_auto)

    if mse_manuell < mse_auto:
        print("\nGlückwusch !! Ihr wart bei der Selektion der Stützpunkte besser, als automatisches Hinfügen nach großtem Modelfehler !!\n")

    #plt.show()



    ################# HAND-ON: Eine unbekannte Funktion (aus breits oben definierten) fitten ##########
    #################  dabei versuchen minimale anzahl an stützpunkten zu nehmen ##########

    #wähle die funktion aus
    f_unbekannte_funktion = f_herbie_3()

    #extrahiere wertebereich
    min_x, max_x, min_y, max_y = f_unbekannte_funktion.getRange()

    n_start_points = 15
    x_p,y_p = getInitPoints("CenteredRandomGridCombinedPointsWithEdges", n_start_points, min_x,max_x, min_y,max_y)

    #berechne die funktionswerte für startpunkte
    z_p = f_unbekannte_funktion.getValue(x_p, y_p)

    #fitte variogram manuell (aus der variogram-aufgabe copy-pasten) oder automatisch mit der folgenden funktion

    best_model, best_score, bin_center, gamma = getBestCovarianceModel(x_p, y_p, z_p, True)

    #berechne kriging-model
    krig = calculateKrigingModel("universal_quadratic", best_model, x_p, y_p, z_p)

    #man kann das feld und sqrt(varianz) = std.abw anschauen
    fig, (ax1, ax2) = plt.subplots(1,2)
    plotKrigModel(krig, min_x, max_x, min_y, max_y, x_p, y_p, fig, ax1, ax2)

    fig, ax1 = plt.subplots()
    se_map, mse_manuell = getSEmapOfKrigModel(krig, f_unbekannte_funktion, min_x, max_x, min_y, max_y, x_p, y_p, fig, ax1)


    ###### diesen schritt wiederholen, bis ihr mitm ergebniss zufrieden seid

    # hinfüge den punkt mit dem größten Modelfehler oder den punkt, der ihr glaubt würde den fit verbessern
    x_new, y_new = getPointWithMaxValue(krig, min_x, max_x, min_y, max_y) #oder manuell festlegen
    z_new = f_bekannte_funktion_2.getValue(x_new, y_new)
    x_p = np.append(x_p, x_new)
    y_p = np.append(y_p, y_new)
    z_p = np.append(z_p, z_new)

    best_model, best_score, bin_center, gamma = getBestCovarianceModel(x_p, y_p, z_p, True)

    # berechne kriging-model
    krig = calculateKrigingModel("universal_quadratic", best_model, x_p, y_p, z_p)

    # man kann das feld und sqrt(varianz) = std.abw anschauen
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plotKrigModel(krig, min_x, max_x, min_y, max_y, x_p, y_p, fig, ax1, ax2)
    plotCovarianceModel(bin_center, gamma, best_model, ax3)





    n_points_to_fit = [15, 20, 25, 30]
    n_runs = 10 #anzahl der wiederholungen mit gleichen parametern (übertreibt nicht)
    init_faction = 0.7 #anteil an startpunkten zum ersten berechnen des models, der rest wird nach dem größten SE hinzugefügt

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
    model_mse, model_mse_std, best_krige, best_cov_model, best_bin_center, best_gamma = analyseFunction(f_current,n_points_to_fit, n_runs, init_faction, ax1, ax2, ax3, ax4)


"""