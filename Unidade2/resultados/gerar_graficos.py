"""
Gera gráficos para as questões 2, 4, 6, 7, 11, 12, 13 da lista de OpenMP.
Execute: python3 gerar_graficos.py
Os PNGs são salvos em ./graficos/
"""

import matplotlib
matplotlib.use('Agg')  # backend sem janela — funciona sem display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Diretório de saída dos gráficos
OUT = os.path.join(os.path.dirname(__file__), "graficos")
os.makedirs(OUT, exist_ok=True)

CORES = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# =============================================================================
# Q2 — Speedup da Regra do Trapézio
# =============================================================================
threads_q2 = [1, 2, 4, 8]
tempo_10M  = [0.0555, 0.0113, 0.0154, 0.0082]
tempo_100M = [0.3626, 0.1899, 0.1362, 0.0551]

speedup_10M  = [tempo_10M[0]  / t for t in tempo_10M]
speedup_100M = [tempo_100M[0] / t for t in tempo_100M]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Q2 — Regra do Trapézio com omp_get_wtime()', fontsize=13, fontweight='bold')

# Tempos
ax = axes[0]
ax.plot(threads_q2, tempo_10M,  'o-', color=CORES[0], linewidth=2, markersize=7, label='n = 10 milhões')
ax.plot(threads_q2, tempo_100M, 's-', color=CORES[1], linewidth=2, markersize=7, label='n = 100 milhões')
ax.set_xlabel('Número de Threads')
ax.set_ylabel('Tempo (segundos)')
ax.set_title('Tempo de Execução')
ax.set_xticks(threads_q2)
ax.legend()
ax.grid(True, alpha=0.35)

# Speedup
ax = axes[1]
ax.plot(threads_q2, speedup_10M,  'o-', color=CORES[0], linewidth=2, markersize=7, label='n = 10 milhões')
ax.plot(threads_q2, speedup_100M, 's-', color=CORES[1], linewidth=2, markersize=7, label='n = 100 milhões')
ax.plot(threads_q2, threads_q2, '--', color='gray', linewidth=1.5, label='Speedup ideal')
ax.set_xlabel('Número de Threads')
ax.set_ylabel('Speedup (T1 / Tp)')
ax.set_title('Speedup vs. Ideal')
ax.set_xticks(threads_q2)
ax.legend()
ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig(f'{OUT}/q2_trap_speedup.png')
plt.close()
print("Salvo: q2_trap_speedup.png")

# =============================================================================
# Q4 — Distribuição de Iterações (escalonamento STATIC)
# =============================================================================
configs = [(8, 2), (8, 4), (16, 4), (32, 8)]
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle('Q4 — Escalonamento Padrão (STATIC) — Distribuição das Iterações',
             fontsize=13, fontweight='bold')

dados_q4 = {
    (8,  2): {0: (0,3),  1: (4,7)},
    (8,  4): {0: (0,1),  1: (2,3),  2: (4,5),  3: (6,7)},
    (16, 4): {0: (0,3),  1: (4,7),  2: (8,11), 3: (12,15)},
    (32, 8): {0: (0,3),  1: (4,7),  2: (8,11), 3: (12,15),
              4: (16,19),5: (20,23),6: (24,27), 7: (28,31)},
}

for idx, (n_iter, n_t) in enumerate(configs):
    ax = axes[idx // 2][idx % 2]
    info = dados_q4[(n_iter, n_t)]
    for t, (ini, fim) in info.items():
        ax.barh(t, fim - ini + 1, left=ini, color=CORES[t % len(CORES)],
                edgecolor='white', height=0.6,
                label=f'Thread {t}: iters {ini}–{fim}')
        ax.text(ini + (fim - ini + 1)/2, t, f'{ini}–{fim}',
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    ax.set_yticks(list(info.keys()))
    ax.set_yticklabels([f'Thread {t}' for t in info.keys()])
    ax.set_xlabel('Iteração')
    ax.set_title(f'{n_iter} iterações, {n_t} threads → chunk = {n_iter // n_t}')
    ax.set_xlim(-0.5, n_iter)
    ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/q4_escalonamento_static.png')
plt.close()
print("Salvo: q4_escalonamento_static.png")

# =============================================================================
# Q6 — Comparação dos tipos de Schedule
# =============================================================================
schedules = ['static', 'static,2', 'dynamic', 'dynamic,2', 'guided']
# Distribuições reais capturadas na execução com n=20 e 4 threads
dist_q6 = {
    'static':    {0:[1,2,3,4,5],    1:[6,7,8,9,10],     2:[11,12,13,14,15], 3:[16,17,18,19]},
    'static,2':  {0:[1,2,9,10,17,18],1:[3,4,11,12,19],  2:[5,6,13,14],      3:[7,8,15,16]},
    'dynamic':   {0:[2,5,8,12,17],  1:[1,7,9,10,13,14,16,18],2:[3,4,6,11,15,19],3:[]},
    'dynamic,2': {0:[],             1:[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],2:[1,2,19],3:[]},
    'guided':    {0:[10,11,12,19],  1:[1,2,3,4,5,13,14,15,16,17],2:[6,7,8,9,18],3:[]},
}

fig, axes = plt.subplots(len(schedules), 1, figsize=(13, 14))
fig.suptitle('Q6 — Distribuição de Iterações por Tipo de Schedule\n'
             '(n=20 trapézios, 4 threads, intervalo [0,1])',
             fontsize=13, fontweight='bold')

for ax, sched in zip(axes, schedules):
    dist = dist_q6[sched]
    for t in range(4):
        iters = dist[t]
        for it in iters:
            ax.barh(t, 1, left=it - 0.5, color=CORES[t],
                    edgecolor='white', height=0.55, alpha=0.88)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([f'Thread {t}' for t in range(4)])
    ax.set_xlim(0, 20)
    ax.set_xlabel('Iteração (1 a 19)')
    ax.set_title(f'schedule({sched})', fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    # Legenda de contagem por thread
    counts = [len(dist[t]) for t in range(4)]
    ax.text(20.2, 1.5, f'Iters/thread: {counts}', fontsize=8.5, va='center')

plt.tight_layout()
plt.savefig(f'{OUT}/q6_schedules.png')
plt.close()
print("Salvo: q6_schedules.png")

# =============================================================================
# Q7 — Atomic em variáveis privadas: comparação de tempo
# =============================================================================
threads_q7 = [1, 2, 4, 8]
# Tempo com 1 thread (baseline medido com num_threads(1))
baseline_1t = [0.3185, 0.3212, 0.3518, 0.2958]
# Tempo com N threads (execução em paralelo)
tempo_Nt    = [0.3110, 0.3301, 0.4294, 0.6130]

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle('Q7 — #pragma omp atomic em Variáveis Privadas\n'
             '(n = 10.000.000, cada thread faz o loop completo)',
             fontsize=12, fontweight='bold')

x = np.arange(len(threads_q7))
w = 0.35
bars1 = ax.bar(x - w/2, baseline_1t, w, label='Baseline (1 thread, num_threads=1)',
               color=CORES[0], alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + w/2, tempo_Nt,   w, label='Execução com N threads',
               color=CORES[2], alpha=0.85, edgecolor='white')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}s', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}s', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Número de Threads')
ax.set_ylabel('Tempo (segundos)')
ax.set_xticks(x)
ax.set_xticklabels([f'{t} thread(s)' for t in threads_q7])
ax.legend()
ax.grid(True, axis='y', alpha=0.35)
ax.set_ylim(0, 0.75)

nota = ('Nota: cada thread executa n=10M iterações em sua\n'
        'própria variável — o tempo cresce com mais threads\n'
        'porque o TRABALHO TOTAL cresce (T×n iterações),\n'
        'não por serialização do atomic.')
ax.text(0.98, 0.97, nota, transform=ax.transAxes, fontsize=8,
        va='top', ha='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUT}/q7_atomic_privado.png')
plt.close()
print("Salvo: q7_atomic_privado.png")

# =============================================================================
# Q11 — Histograma Paralelo: Speedup
# =============================================================================
threads_q11 = [1, 2, 4, 8]
tempo_q11   = [0.0121, 0.0078, 0.0057, 0.0088]
speedup_q11 = [tempo_q11[0] / t for t in tempo_q11]
efic_q11    = [s / t for s, t in zip(speedup_q11, threads_q11)]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle('Q11 — Histograma Paralelo (n=10M elementos, 20 buckets)', fontsize=13, fontweight='bold')

ax = axes[0]
ax.plot(threads_q11, speedup_q11, 'o-', color=CORES[0], linewidth=2, markersize=8, label='Speedup real')
ax.plot(threads_q11, threads_q11, '--', color='gray', linewidth=1.5, label='Ideal')
for t, s in zip(threads_q11, speedup_q11):
    ax.annotate(f'{s:.2f}x', (t, s), textcoords='offset points', xytext=(5, 5), fontsize=9)
ax.set_xlabel('Número de Threads')
ax.set_ylabel('Speedup')
ax.set_title('Speedup')
ax.legend()
ax.set_xticks(threads_q11)
ax.grid(True, alpha=0.35)

ax = axes[1]
bars = ax.bar(threads_q11, [t*1000 for t in tempo_q11], color=CORES[:4], edgecolor='white', alpha=0.85)
for bar, t in zip(bars, tempo_q11):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{t*1000:.1f}ms', ha='center', va='bottom', fontsize=9)
ax.set_xlabel('Número de Threads')
ax.set_ylabel('Tempo (ms)')
ax.set_title('Tempo de Execução')
ax.set_xticks(threads_q11)
ax.grid(True, axis='y', alpha=0.35)

plt.tight_layout()
plt.savefig(f'{OUT}/q11_histograma_speedup.png')
plt.close()
print("Salvo: q11_histograma_speedup.png")

# =============================================================================
# Q12 — Monte Carlo π: Precisão e Speedup
# =============================================================================
ns = [1_000_000, 10_000_000, 100_000_000]
threads_q12 = [1, 2, 4, 8]

tempos_q12 = {
    1: [0.0180, 0.1854, 1.8945],
    2: [0.0088, 0.0982, 0.9761],
    4: [0.0060, 0.0530, 0.5586],
    8: [0.0099, 0.0354, 0.3771],
}
erros_q12 = {
    1: [0.00097535, 0.00026585, 0.00007421],
    2: [0.00091135, 0.00019335, 0.00000761],
    4: [0.00194335, 0.00031455, 0.00001111],
    8: [0.00075935, 0.00042735, 0.00006957],
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Q12 — Monte Carlo para estimar π', fontsize=13, fontweight='bold')

# Speedup por n (n = 100M)
ax = axes[0]
speedup_ref = tempos_q12[1][2]
speedup_vals = [speedup_ref / tempos_q12[t][2] for t in threads_q12]
ax.plot(threads_q12, speedup_vals, 'o-', color=CORES[0], linewidth=2, markersize=8)
ax.plot(threads_q12, threads_q12, '--', color='gray', label='Ideal')
for t, s in zip(threads_q12, speedup_vals):
    ax.annotate(f'{s:.2f}x', (t, s), textcoords='offset points', xytext=(4, 4), fontsize=9)
ax.set_xlabel('Threads')
ax.set_ylabel('Speedup')
ax.set_title('Speedup (n = 100 milhões)')
ax.set_xticks(threads_q12)
ax.legend()
ax.grid(True, alpha=0.35)

# Erro por número de lançamentos (4 threads)
ax = axes[1]
ns_labels = ['1M', '10M', '100M']
for t, cor in zip(threads_q12, CORES):
    ax.plot(ns_labels, erros_q12[t], 'o-', color=cor, linewidth=2, markersize=7, label=f'{t} thread(s)')
ax.set_xlabel('Número de Lançamentos')
ax.set_ylabel('Erro absoluto |π_est − π|')
ax.set_title('Precisão vs. Lançamentos')
ax.legend()
ax.grid(True, alpha=0.35)
ax.set_yscale('log')

# Tempo por threads (n = 100M)
ax = axes[2]
tempos_100M = [tempos_q12[t][2] for t in threads_q12]
bars = ax.bar(threads_q12, tempos_100M, color=CORES[:4], edgecolor='white', alpha=0.85)
for bar, t in zip(bars, tempos_100M):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{t:.3f}s', ha='center', va='bottom', fontsize=9)
ax.set_xlabel('Threads')
ax.set_ylabel('Tempo (s)')
ax.set_title('Tempo (n = 100 milhões)')
ax.set_xticks(threads_q12)
ax.grid(True, axis='y', alpha=0.35)

plt.tight_layout()
plt.savefig(f'{OUT}/q12_montecarlo_pi.png')
plt.close()
print("Salvo: q12_montecarlo_pi.png")

# =============================================================================
# Q13 — Count Sort: Serial vs Paralelo vs qsort
# =============================================================================
ns_q13 = [1000, 3000, 5000, 8000, 10000]
threads_q13 = [1, 2, 4, 8]

serial_q13 = [0.0048, 0.0540, 0.1323, 0.3267, 0.5264]
qsort_q13  = [0.0003, 0.0004, 0.0007, 0.0017, 0.0020]
paralelo_q13 = {
    1: [0.0052, 0.0495, 0.1268, 0.3199, 0.5086],
    2: [0.0032, 0.0226, 0.0650, 0.1646, 0.2516],
    4: [0.0020, 0.0152, 0.0420, 0.0787, 0.1312],
    8: [0.0141, 0.0263, 0.0487, 0.0585, 0.1043],
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Q13 — Count Sort: Serial vs Paralelo vs qsort', fontsize=13, fontweight='bold')

# Tempo por n
ax = axes[0]
ax.plot(ns_q13, serial_q13, 'k--o', linewidth=2, markersize=7, label='Count Sort serial')
ax.plot(ns_q13, qsort_q13,  's-',  color='gray', linewidth=2, markersize=7, label='qsort (stdlib)')
for t, cor in zip(threads_q13, CORES):
    ax.plot(ns_q13, paralelo_q13[t], 'o-', color=cor, linewidth=2, markersize=7, label=f'Paralelo {t}t')
ax.set_xlabel('Tamanho do Array (n)')
ax.set_ylabel('Tempo (segundos)')
ax.set_title('Tempo de Execução')
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.35)

# Speedup paralelo vs serial (n = 10000)
ax = axes[1]
speedups_q13 = [serial_q13[-1] / paralelo_q13[t][-1] for t in threads_q13]
bars = ax.bar([str(t) for t in threads_q13], speedups_q13, color=CORES[:4], edgecolor='white', alpha=0.85)
for bar, s in zip(bars, speedups_q13):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f'{s:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.2, label='Sem speedup')
ax.set_xlabel('Número de Threads')
ax.set_ylabel('Speedup vs Count Sort Serial')
ax.set_title('Speedup (n = 10.000)')
ax.legend()
ax.grid(True, axis='y', alpha=0.35)

nota_q = (f'qsort (n=10000): {qsort_q13[-1]*1000:.1f}ms\n'
          f'Count sort 8t:   {paralelo_q13[8][-1]*1000:.1f}ms\n'
          f'qsort é {paralelo_q13[8][-1]/qsort_q13[-1]:.0f}× mais rápido que paralelo')
ax.text(0.98, 0.97, nota_q, transform=ax.transAxes, fontsize=8.5,
        va='top', ha='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUT}/q13_count_sort.png')
plt.close()
print("Salvo: q13_count_sort.png")

print("\nTodos os gráficos gerados em:", OUT)
