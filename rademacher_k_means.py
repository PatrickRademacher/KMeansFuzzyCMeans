#Patrick Rademacher
#Anthony Rhodes
#Machine Learning
#Portland State University
#June 3, 2020
#Program 3

import math
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import linalg as LA
from matplotlib import cm
from collections import OrderedDict
import colorsys
import logging
import unicodedata as u
from matplotlib.backends.backend_pdf import PdfPages
regular_k_means = PdfPages('regular_k.pdf')
fuzzzy = PdfPages('fuzzzy_k.pdf')
plt.rcParams['axes.facecolor'] = [0.09, 0.07, 0.08]
plt.rcParams['savefig.facecolor'] = "0.8"
from sklearn.utils import shuffle




def initialize_k_vals(points, random_starters):
    for i in range(K):
        cutoff = len(points) - 1
        val = random.randint(0, cutoff)
        while (val in random_starters[i]):
            val = random.randint(0, cutoff)
        kx.append(points[val][0])
        ky.append(points[cutoff-val][1])
        new_random = [points[val][0], points[cutoff-val][1]]
        random_starters[i] = new_random
    return points, random_starters

def first_iteration(points, k_means, clusters, K, printer, ax_array):
    to_return = []
    art = None
    for point in points:
        indexx = 0
        min_distance = pow((k_means[0][0]-point[0]),2)+pow((k_means[0][1]-point[1]), 2)
        for i in range(0, K):
            check = pow((k_means[i][0]-point[0]),2)+pow((k_means[i][1]-point[1]), 2)
            if check < min_distance:
                min_distance = check
                indexx = i
        forall = [point, indexx] 
        to_return.append(forall)
        clusters[indexx].append(point)
    if printer == True:
        ax_array = np.ravel(ax_array)
        ax = ax_array[0]
        art = plot_data(clusters, K, k_means, ax)
    for c in range(len(clusters)):
        avg_x = 0
        avg_y = 0
        for l in range(len(clusters[c])):
            avg_x += clusters[c][l][0]
            avg_y += clusters[c][l][1]
        if len(clusters[c]) == 0:
            k_means[c][0] = 0
            k_means[c][1] = 0
        else:
            k_means[c][0] = avg_x/len(clusters[c])
            k_means[c][1] = avg_y/len(clusters[c])
    plt.tight_layout()
    return to_return, clusters, k_means, art

def cycle_iterations(pv, clusters, k_means, K, printer, ax_array):
    max_iterations = 8
    total_iterations = 0
    art = None
    for q in range(max_iterations):
        p = 0
        for j in range(len(pv)):
            index = pv[j][1]
            point = pv[j][0]
            new_index = -1
            min_distance = pow((k_means[0][0]-point[0]),2)+pow((k_means[0][1]-point[1]), 2)
            for i in range(1, K):
                check = pow((k_means[i][0]-point[0]),2)+pow((k_means[i][1]-point[1]), 2)
                if check < min_distance:
                    min_distance = check
                    new_index = i 
            if new_index != index:
                clusters[index].remove(point)
                clusters[new_index].append(point)  
                #clusters[index].remove(point)
                pv[j][1] = new_index
            else:
                p += 1
        if printer == True:
            ax_array = np.ravel(ax_array)
            ax = ax_array[q+1]
            art = plot_data(clusters, K, k_means, ax)
        
        total_iterations += 1
        for c in range(len(clusters)):
            avg_x = 0
            avg_y = 0
            for l in range(len(clusters[c])):
                avg_x += clusters[c][l][0]
                avg_y += clusters[c][l][1]
            if len(clusters[c]) != 0:
                k_means[c][0] = avg_x/len(clusters[c])
                k_means[c][1] = avg_y/len(clusters[c])
    sum_squared_dist = 0
    making_sure = 0
    for c in range(len(clusters)):
        for l in range(len(clusters[c])):
            sum_squared_dist += pow((clusters[c][l][0] - k_means[c][0]),2) +  pow((clusters[c][l][1] - k_means[c][1]),2)
            avg_y += clusters[c][l][1]
            k_means[c][0] = avg_x/len(clusters[c])
            k_means[c][1] = avg_y/len(clusters[c])
    return sum_squared_dist, clusters, k_means, art
        

def plot_data(clusters, K, k_means, ax):
    
    color_corrector = [.92,.96,.91,.94,.99,.93,.95,.97] #= 0.71 #(random.randint(0,99)) * .01
    count = 0
    kdot = ['k\u2080', 'k\u2081', 'k\u2082', 'k\u2083', 'k\u2084', 'k\u2085', 'k\u2086', 'k\u2087']
    toreturn = [[]*2]
    for cluster in clusters:
        if count != 1:
            rgb_values = [((color_corrector[i]/K)*(i+color_corrector[i])) for i in range(K)]
        if K >2:
            rgbinv =  [1 - rgb_values[2], 1 - rgb_values[0], 1 - rgb_values[2]]
        else:
             rgbinv =  [1 - rgb_values[1], 1 - rgb_values[0], 1 - rgb_values[1]]
        if count > 0:
            cluster_color = rgb_values[clusters.index(cluster)]
        else:
            cluster_color = rgb_values[clusters.index(cluster)]
        thecolorr = twilight(cluster_color)
        thecolor2 = colorsys.rgb_to_hsv(thecolorr[0], thecolorr[1], thecolorr[2])
        thecolorr = (thecolor2[0], thecolor2[1], thecolor2[2], thecolorr[3])
        start = .45
        stop = .35
        thecolor = []
        h = 0
        while h < len(cluster):
            thecolor.append((thecolorr[0], thecolorr[1], thecolorr[2], stop))
            h += 1
        d = 0
        logger = logging.getLogger()
        old_level = logger.level
        logger.setLevel(100)
        for percluster in cluster:
            toreturn.append(ax.scatter(percluster[0], percluster[1], c = thecolor[d], cmap=twilight, s = .2))
            #toreturn.append(ax.set_ylabel("K" + str(d), color = rgb_values[d], fontsize = 3))
            d = d + 1
        
        toreturn.append(ax.text(k_means[count][0], k_means[count][1], kdot[count], color=[113/255, 248/255,184/255], fontsize=8))
        coords = ""
        #for e in range(0, K):
        #    coords += "K = (" + str(k_means[e][0])[0:6] + ", " + str(k_means[e][1])[0:6] + ")\n"
        toreturn.append(ax.set_xlabel(coords, fontsize = 6, color = 'black'))
        count += 1 
        logger.setLevel(old_level)
    return toreturn

def update_k_means(w, K, k_means, p, m):
    for g in range(K):
        numeratorx = 0
        numeratory = 0
        denominator = 0
        xs = []
        ys = []
        po = []
        for h in range(1500):
            po.append(w[h][g] * w[h][g])
            xs.append(p[h][0])
            ys.append(p[h][1])
        po = np.array(po)
        xs = np.array(xs)
        ys = np.array(ys)
        xs = np.multiply(xs, po)
        ys = np.multiply(ys, po)
        ys = np.array(ys)
        po = np.array(po)
        numeratorx = np.sum(xs)
        numeratory = np.sum(ys)
        denominator = np.sum(po)
        k_means[g][0] = np.divide(numeratorx, denominator)
        k_means[g][1] = np.divide(numeratory, denominator)
    return(k_means)


def update_weights(w, K, k_means, p, m):
    same_size_arrays = []  
    new_weights = np.zeros(np.shape(w)) 
    for s in range(K):
        same_size_arrays.append([k_means[s] for i in range(1500)])
    same_size_arrays = np.array(same_size_arrays)
    for i in range(1500):
        for j in range(K):
            sum_denominator = 0
            numerator = LA.norm(p[i] - k_means[j])
            for k in range(K):
                denominator = LA.norm(p[i] - k_means[k])
                total_denominator = (numerator/denominator)
                total_denominator = np.array(total_denominator)
                total_denominator = np.power(total_denominator, (2/(m-1)))
                sum_denominator += total_denominator
            new_weights[i][j] = np.divide(1, sum_denominator)
    return new_weights


        
'''
all_ss = []
clusterz = []
meanz = []
final_k_ss = []
figgies = [0, 0, 0, 0, 0, 0, 0, 0]
for K in range(2, 9):
    sum_squared_distance = []
    minx, maxx, miny, maxy = 99999, -100000, 99999, -100000
    graph_best = []
    random_starters = [[[] for j in range(K)]for w in range(10)]
    figgies[K-2], ax_array = plt.subplots(3, 3, sharex=False, sharey=True, constrained_layout = False, figsize = (9.5, 7))
    mngr = plt.get_current_fig_manager()
    geom = mngr.window.geometry()
    x,y,dx,dy = geom.getRect()
    #print(dx, dy)
    mngr.window.setGeometry(x, y, dx, dy)
    for a in range(0, 10):
        print('a = ' + str(a))
        good_to_print = False
        twilight = cm.get_cmap('twilight', 200)
        r = 0
        datafile = open("cluster_dataset.txt", "r")
        points = []
        setp = []
        xvals = []
        yvals = []
        c = 0
        clusters = [[] for i in range(K)]
        kx = []
        ky = []
        for datas in datafile.read().split('\n'):
            for dataz in datas.split(' '):
                if len(dataz) > 1:
                    setp.append(float(dataz))
                    c += 1
                    if c == 2:
                        points.append(setp)
                        xvals.append(setp[0])
                        yvals.append(setp[1])
                        setp = []
                        c = 0
        minx = min(xvals)
        maxx = max(xvals)
        maxy = max(yvals)
        miny = min(yvals)
        total_range = abs(maxx - minx)
        converter = 1/total_range
        #print(converter)
        if a < 9:
            ax_array = np.ravel(ax_array)
            axx = ax_array[a]
            axx.set_xlim(minx -  .5, 3)
            axx.set_ylim(miny -  .5, maxy + 2.25)
            
            #axx.axhspan(maxy, maxy+2.25, alpha = 1)
            #l, b, w, h = axx.get_position().bounds
            #l = l - .2
            #axx.set_position
        if a != 9:
            points, random_starters[a] = initialize_k_vals(points, random_starters[a])
            k_means = random_starters[a].copy()
            #print(k_means)
        else:
            good_to_print = True
            k_means = graph_best
        pv, clusters, k_means, art1 = first_iteration(points, k_means, clusters, K, good_to_print, np.ravel(ax_array))
        #for cluster in clusters:
            #print(len(cluster))
        ss, clust, km, art2 = cycle_iterations(pv, clusters, k_means, K, good_to_print, np.ravel(ax_array))
        if a == 9:
            figgies[K-2].suptitle("Regular K-Means:   K = "+  str(K)+ ", Sum of Squares = " +str(round(final_k_ss[K-2], 2)),  color = [113/255, 208/255,184/255], y=.93, fontsize = 17, fontweight = 'bold')
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            plt.savefig(regular_k_means, format='pdf')
            plt.show()
            
            
        else:
            sum_squared_distance.append(float(ss))
            clusterz.append(clust)
            meanz.append(km)
            datafile.close()
            if a == 8:
                ssmin = 99999999999
                for z in range(0,9):
                    #print(sum_squared_distance[z])
                    if float(sum_squared_distance[z]) < ssmin:
                        ssmin = sum_squared_distance[z]
                        r = z
                final_k_ss.append(ssmin)
                graph_best = random_starters[r]
                all_ss.append(sum_squared_distance)

print(final_k_ss)
for j in range(len(all_ss)):
    print(all_ss[j])
regular_k_means.close()
plt.show()

'''
plt.rcParams['savefig.facecolor'] = "0.8"

figgies = [0, 0, 0, 0, 0, 0, 0]
all_ss = [[] for tt in range(7)]
print(all_ss)
keep_data = []
m = 2
for K in range(2, 9):
    finder = -1
    remember_weights = []
    s_count_list = []
    sum_squared_distance = []
    graph_best = []
    random_starters = [[[] for j in range(K)]for w in range(10)]
    figgies[K-2], ax_array = plt.subplots(3, 3, sharex=True, sharey=True, constrained_layout = True, figsize = (9.5,7))
    m = 2
    ss = [[] for t in range(9)]
    art = []
    for a in range(0, 10):
        clusterindex = -1
        good_to_print = False
        twilight = cm.get_cmap('twilight', 200)
        r = 0
        datafile = open("cluster_dataset.txt", "r")
        points = []
        setp = []
        xvals = []
        yvals = []
        c = 0
        clusters = [[] for i in range(K)]
        kx = [0 for g in range(K)]
        ky = [0 for q in range(K)]
        for datas in datafile.read().split('\n'):
            for dataz in datas.split(' '):
                if len(dataz) > 1:
                    setp.append(float(dataz))
                    c += 1
                    if c == 2:
                        points.append(setp)
                        xvals.append(setp[0])
                        yvals.append(setp[1])
                        setp = []
                        c = 0
        minx = min(xvals)
        maxx = max(xvals)
        maxy = max(yvals)
        miny = min(yvals)
        total_range = abs(maxx - minx)
        converter = 1/total_range
        if a < 9:
            ax_array = np.ravel(ax_array)
            axx = ax_array[a]
            axx.set_xlim(minx -  .5, 3)
            axx.set_ylim(miny -  .5, maxy + 2.25)
            k_means = [[0, 0] for x in range(K)]
            initial_weights =  [[] for v in range(1500)]
            clusters = [[] for h in range(K)]
            points, random_starters[a] = initialize_k_vals(points, random_starters[a])
            k_means = random_starters[a].copy()
            beat_ss = 0
            for g in range(1500):
                k_means = random_starters[a].copy()
                o=0
                total = 0
                for f in range(K):
                    if f == K - 1:
                        initial_weights[g].append(1000 - total)
                        #print(ww[g,f])
                    else:
                        cool =  random.randint(0, ((int(1000/K)) + o))
                        cool = cool
                        initial_weights[g].append(cool)
                        o += (int(1000/K)) - cool
                        total += cool
            initial_weights = np.array(initial_weights)
            initial_weights = np.divide(initial_weights, 1000)
            initial_weights = shuffle(initial_weights)
            remember_weights.append(initial_weights)
            k_means = np.array(k_means)
            updated_w = initial_weights.copy()
            for z in range(9):
                checker = np.copy(update_weights(updated_w, K, k_means, points, m))
                updated_w = checker
                k_means = update_k_means(updated_w, K, k_means, points, m)
            
        if a == 9:
            k_means = random_starters[finder].copy()
            initial_weights = remember_weights[finder].copy()
            print(finder)
            clusters = [[] for d in range(K)]
            points = np.array(points)
            for z in range(9): 
                clusters = [[] for d in range(K)]
                for tt in range(1500):
                    max_v = 0
                    for t in range(K):
                        if updated_w[tt][t] > max_v:
                            max_v = updated_w[tt][t]
                            marker = t
                    clusters[marker].append(points[tt])  
                ax_array = np.ravel(ax_array)
                ax = ax_array[z]
                art.append(plot_data(clusters, K, k_means, ax))
                checker = np.copy(update_weights(updated_w, K, k_means, points, m))
                updated_w = checker
                k_means = update_k_means(updated_w, K, k_means, points, m)
            figgies[K-2].suptitle("Fuzzy C-Means:    K = "+  str(K)+ ", Sum of Squares = " +str(round(min(ss), 2)),  color = [113/255, 208/255,184/255], y=.93, fontsize = 17, fontweight = 'bold')
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            plt.savefig(fuzzzy, format='pdf')
            plt.show()
               
        else:  
            for i in range(1500):
                minimum = 9999999
                for j in range(K):
                    best_sd_maybe = pow(LA.norm(points[i] - k_means[j]), 2)
                    if best_sd_maybe < minimum:
                        minimum = best_sd_maybe
                        order_num = j
                clusters[order_num].append(points[i])
                beat_ss += minimum
            ss[a] = beat_ss
            if a == 8:
                finder = ss.index(min(ss))
                k_means = random_starters[finder]
                initial_weights = remember_weights[finder].copy()
                print("finder = " + str(finder))
                print(ss)
                all_ss[K-2] = min(ss) 
        
        
         
                
fuzzzy.close()    
        


