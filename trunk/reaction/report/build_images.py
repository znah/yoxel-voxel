# -*- coding: utf-8 -*-

from numpy import *
import os
import matplotlib.pyplot as pl

#pl.rc('text', usetex=True)
pl.rc('font', family='arial')

def make_count_chart(inName, outName):
    a = fromfile(inName, int32).reshape(768, 1024)
    
    fig = pl.figure()
    fig.subplots_adjust(left = 0.05, right=1.05, bottom = 0.05, top=0.95, wspace = 0)
    fig.set_figwidth(9)

    ax = fig.add_subplot(111)

    cax = ax.imshow(a, origin='bottom', interpolation='nearest', cmap=pl.cm.gray_r)

    fig.colorbar(cax)
    
    pl.savefig(outName, dpi=150)
    

def loadlog(fn):
    with file(fn) as f:
        s = f.read()
    ss = s.split('--\n')
    log = []
    for item in ss:
        fs = [ s.split() for s in item.split('\n') if len(s) > 0 ]
        fs = [ (f[0], tuple(map(float, f[1:]))) for f in fs ]
        d = dict( fs )
        if len(d) > 0:
            log.append(d)
    return log



def make_trace_res_time_chart():
    fig = pl.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left = 0.08, right=0.95, bottom = 0.1, top=0.95)

    def plot_t_rc(fn, label, style='o-'):
        log = loadlog(fn)
        dt = [v['prof.traceTime'] for v in log]
        rayCount = [prod(v['resolution']) for v in log]
        return pl.plot(array(rayCount)*1e-6, dt, style, label = label)

    plot_t_rc('data/trace_simple.log' , u'Best'   )
    plot_t_rc('data/trace_shuffle.log', u'Shuffle rays')
    plot_t_rc('data/trace_shared.log' , u'Shared stack' )
    plot_t_rc('data/trace_notex.log'  , u'No texture fetch' )
    pl.legend(loc='best')
    
    pl.xlabel(u'Количество лучей (млн.)')
    pl.ylabel(u'Время (мс)')

    pl.savefig('images/trace_res_time.pdf')


def make_trace_lod_time_chart():
    fig = pl.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left = 0.08, right=0.95, bottom = 0.1, top=0.95)

    log = loadlog('data/trace_lod_512.log')
    dt = array([v['prof.traceTime'] for v in log])
    lodLimit = array([prod(v['lodLimit']) for v in log])
    pl.plot(lodLimit, dt, 'o-', label='512x384' )

    log = loadlog('data/trace_lod_1024.log')
    dt = array([v['prof.traceTime'] for v in log])
    lodLimit = array([prod(v['lodLimit']) for v in log])
    pl.plot(lodLimit, dt, 'o-', label='1024x768' )


    pl.xlim([lodLimit.min()-1, lodLimit.max() + 1])
    pl.ylim([0.0, dt.max() + 1.0])
    pl.xlabel(u'Глубина дерева')
    pl.ylabel(u'Время (мс)')
    pl.legend(loc='best')
    
    pl.savefig('images/trace_lod_time.pdf')

def make_trace_enter_exit():
    def loaddmp(fn):
        a = fromfile(fn, int32)
        a.shape = (768, 1024)
        return a
        
    def plot_stats(path):
        t1 = loaddmp(path+'/enter_dump.dat')[::4, ::8].ravel()
        t2 = loaddmp(path+'/exit_dump.dat')[::4, ::8].ravel()

        o = argsort(t1)
        t1 = t1[o]
        t2 = t2[o]
        
        pl.plot(t1, 'o')
        pl.plot(t2, 'o', color='red')

    fig = pl.figure(figsize=(8, 4))
    fig.subplots_adjust(left = 0.08, right=0.95, bottom = 0.1, top=0.95)
    
    
    ax1 = fig.add_subplot(121)
    plot_stats('data/dumps')

    ax2 = fig.add_subplot(122, sharey=ax1)
    plot_stats('data/dumps/shuffle')
    pl.ylim([3e7, 3.5e7])
    
    w = 1000
    ax1.set_xlim([11000, 11000+w])
    ax2.set_xlim([3900, 3900+w])
    
    ax1.set_title('coherent')
    ax2.set_title('shuffled')
    
    ax1.set_xlabel(u'warp')
    ax2.set_xlabel(u'warp')
    ax1.set_ylabel(u'время (такты)')
    ax2.set_ylabel(u'время (такты)')
    
    pl.savefig('images/trace_scheduler.pdf')
    

def make_voxel_time():
    logs = load('data/grow_log.npz')
    fig = pl.figure()
    fig.subplots_adjust(left = 0.08, right=0.95, bottom = 0.1, top=0.95)
    
    a = logs['grow.calcAbsorb.voxelize_gl']
    dt, vn, fn = a.T

    pl.plot(vn, dt)
    n = 65536
    pl.vlines(n, 0, 14, linestyle='--')
    pl.text(n+1000, 10, str(n))
    pl.xlabel(u'Количество вершин')
    pl.ylabel(u'Время (мс)')
    pl.ylim([0, 14])
    
    pl.savefig('images/voxel_time.pdf')
    
    
def make_coralgrow_time():
    logs = load('data/grow_log.npz')
    fig = pl.figure()
    fig.subplots_adjust(left = 0.08, right=0.95, bottom = 0.1, top=0.95)
    
    # ['grow.calcAbsorb.voxelize_gl', 
    #  'grow.calcAbsorb.PrepareDiffusionVolume_cu', 
    #  'grow.calcAbsorb.Diffusion_cu', 
    #   grow.growMesh', 'grow']
    
    
    totalTime = logs['grow'][:,0]
    x = r_[:len(totalTime)]
    growMeshTime = logs['grow.growMesh'][:,0]
    diffusionTime = logs['grow.calcAbsorb.Diffusion_cu'][:,0]
    otherTime = totalTime-growMeshTime-diffusionTime
    
    pl.plot(totalTime, 'k-', label=u'Полное время')
    pl.plot(growMeshTime, 'k:.', label=u'Обновление полигональной модели')
    pl.plot(diffusionTime, 'k:', label=u'Диффузия (50 итераций)')
    pl.plot(otherTime, 'k-.', label=u'Прочее')
    
    pl.ylim(0, 800)
    
    pl.legend(loc='best')   
    pl.xlabel(u'Итерация')
    pl.ylabel(u'Время (мс)')
    
    pl.savefig('images/coral_grow_time.pdf')
    

make_count_chart('data/dumps/iter_count.dat', 'images/trace_iters.png') 
os.system('gm convert -quality 90 images/trace_iters.png images/trace_iters.jpg')

make_trace_res_time_chart()
make_trace_lod_time_chart()
make_trace_enter_exit()
make_voxel_time()

make_coralgrow_time()

