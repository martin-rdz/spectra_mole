#! /usr/bin/env python3
# coding=utf-8

"""
Author: radenz@tropos.de

visualisation of spectra
"""
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from . import helpers as h

def mom2str(l, sep=' '):
    return sep.join(['Z:{0:.2f} v:{1:.2f} w:{2:.2f} snr:{5:.2f} | '.format(*mom) for mom in l])

def mom2str_slim(l, sep=' | '):
    return sep.join(['{0:5.1f} {1:5.2f} {2:4.2f} {5:5.1f}'.format(*mom) for mom in l])

def coord_pattern_child(p):
    """the coordinate pattern required for filtering the dictionary for children"""
    return lambda d: d['coords'][:-1] == p and len(d['coords']) == len(p)+1

def iterchilds(travtree, parentcoord):
    for n in list(filter(coord_pattern_child(parentcoord), travtree.values())):
        yield n
        yield from iterchilds(travtree, n['coords'])

def iternodes(travtree):
    level_no=0
    nodes = list(filter(lambda d: len(d['coords']) == level_no+1, travtree.values()))
    for n in nodes:
        yield n
        yield from iterchilds(travtree, n['coords'])

def travtree2text(travtree, show_coordinats=True):
    lines = []
    levels = max(list(map(lambda v: len(v['coords']), travtree.values())))
    if show_coordinats:
        header = ' coordinates '+levels*'  '+ '                     Z       v    width    sk   LDR     t   LDRmax  prom'
    else:
        header = ' '+levels*'  '+ '             Z       v    width    sk   LDR     t   LDRmax  prom'
    lines.append(header)
    
    for v in iternodes(travtree):

        coords = '{:20s}'.format(str(v['coords']))
        bounds = '({:>3d}, {:>3d})'.format(*v['bounds'])
        sp_before = (len(v['coords'])-1)*'  '
        sp_after = (levels-len(v['coords']))*'  '

        mom1 = '{:> 6.2f}, {:> 6.2f}, {:>4.2f}'.format(h.lin2z(v['z']), v['v'], v['width'])
        mom2 = '{:> 3.2f}, {:> 5.1f}, {:> 5.1f}, {:> 5.1f}, {:> 5.1f}'.format(
            v['skew'], h.lin2z(v['ldr']), h.lin2z(v['thres']), h.lin2z(v['ldrmax']), h.lin2z(v['prominence']))
        #mom2 = '{:> 3.2f}, {:> 5.1f}, {:> 5.1f}'.format(v['skew'], h.lin2z(v['ldr']), h.lin2z(v['thres']))
        #txt = "{:>2d}{}{}{}{} {}\n{}{}".format(k, sp_before, bounds, sp_after, mmv, mom1, 33*' ', mom2)
        if show_coordinats:
            txt = "{}{} {}{} {}, {}".format(sp_before, bounds, coords, sp_after, mom1, mom2)
        else:
            txt = "{}{} {} {}, {}".format(sp_before, bounds, sp_after, mom1, mom2)
        lines.append(txt)
    return '\n'.join(lines)


def plot_spectrum(spectra, further_text=None, savepath=None):
    """ 
    new spectrum plotting interface based on list input
    """

    fig, ax = plt.subplots(1, figsize=(7, 4.5), sharex=True)
    #ax.hlines(h.lin2z(valid_LDR), -10, 10, color='grey')

    styles = {'cr': {'color': 'red'},
              'rwp': {'color': 'green'},
              'fit': {'color': 'blue'},
              'cr_broad': {'color': 'red', 'linestyle': ':'},
              'rwp_corr': {'color': 'orange'}
             }
    textprops = {
        'cr': {},
        'rwp': {'xcoord': 0.20},
        'fit': {'xcoord': 0.50},
        'cr_broad': {'xcoord': 0.78},
        'rwp_corr': {}
    }
    
    moment_display = False
    for spectrum in spectra:
        if spectrum['system'] in styles.keys():
            style = styles[spectrum['system']]
            textprop = textprops[spectrum['system']]
        else:
            style = {'color': 'dimgrey'}
            textprop = {}

        if 'moments' in spectrum.keys():
            moment_display = True
            ax.axvline(spectrum['moments'][0].v, ls='dashed',
                       color=style['color'], linewidth=1.2)
            if 'xcoord' in textprop.keys():
                ax.text(textprop['xcoord'], 0.15, mom2str_slim(spectrum['moments'][:4], '\n'), 
                        color=style['color'], fontsize=10, 
                        verticalalignment='top', horizontalalignment='center',
                        family='monospace', transform=fig.transFigure)

        ax.step(spectrum['vel'], h.lin2z(spectrum['specZ']), 
                linewidth=1.5, where='mid', **style, label='Z_'+ spectrum['system'])
        if 'specLDR' in spectrum.keys():
            ax.step(spectrum['vel'], h.lin2z(spectrum['specLDR']), 
                    linewidth=1.5, color='turquoise', where='mid', label='LDR_'+ spectrum['system'])
        if 'specZcx' in spectrum.keys():
            ax.step(spectrum['vel'], h.lin2z(spectrum['specZcx']), 
                    linewidth=1.5, color='crimson', where='mid', label='Zcx_'+ spectrum['system'])
            valid_LDR = np.ma.masked_where(spectrum['specLDRmasked_mask'], spectrum['specZcx'])
            ax.step(spectrum['vel'], h.lin2z(valid_LDR), 
                linewidth=1.5, color='blue', where='mid', label='valid LDR')
        if 'decoupling' in spectrum.keys():
            decoupling_threshold = h.z2lin(h.lin2z(spectrum['specZ'])-spectrum['decoupling'])
            ax.step(spectrum['vel'], h.lin2z(decoupling_threshold), 
                    linewidth=1.5, color='grey', where='mid', label='decoupling')

        # if 'est_meanvel' in spectrum.keys():
        #     ax.axvline(spectrum['est_meanvel'], ls='dashed',
        #            color=style['color'], linewidth=1.2)
        if spectrum['system'] == 'rwp':
            ax.hlines(h.lin2z(spectrum['noise_thres']), -10, 10, 
                      color='grey', linewidth=1.2)

    if further_text is not None:
        s = '; '.join(['{}: {}'.format(k, v) for k, v in further_text.items()])
        ax.text(0.83, 0.20, s, 
            color='black', fontsize=10, 
            verticalalignment='top', horizontalalignment='center',
            family='monospace', transform=fig.transFigure)

    ax.set_xlim([-6,3])
    ax.set_xlim([-11,11])
    ax.set_ylabel('Spectral Reflectivity [dBZ (m s$\\mathregular{^{-1}}$)$\\mathregular{^{-1}}$]', fontsize=13, fontweight='semibold')
    ax.set_xlabel('Velocity [m s$\\mathregular{^{-1}}$]', fontsize=13, fontweight='semibold')
    dt = h.ts_to_dt(spectrum['ts'])
    dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
    ax.set_title('{}   {:>5.0f} m'.format(dt_str, spectrum['range']), fontsize=13, fontweight='semibold')
    ax.set_ylim([-70, 10])


    # custom style
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='both', direction='in',
                   top=True, right=True)
    ax.tick_params(axis='both', which='major',
                   length=5, width=1.5, labelsize=12)
    ax.tick_params(axis='both', which='minor',
                   length=2.5, width=1.5)

    ax.legend()
    plt.tight_layout()
    if moment_display:
        fig.subplots_adjust(bottom=0.28, top=0.98)
    
    if savepath is not None:
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        savename = '{}_{:0>5.0f}m_spectrum.png'.format(dt.strftime('%Y-%m-%d_%H%M%S'), spectrum['range'])
        print('save at ', savepath+savename)
        fig.savefig(savepath + savename, dpi=250)

    return fig, ax