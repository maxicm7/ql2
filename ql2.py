import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Necesitas instalar deap si no lo tienes: pip install deap
from deap import base, creator, tools, algorithms

# ----------------------- Configuración Inicial DEAP -----------------------
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

# ----------------------- Funciones de Carga y Procesamiento de Datos -----------------------
@st.cache_data
def load_data_and_counts(uploaded_file):
    if uploaded_file is None: return None, {}, {}, {}, [], {}, 0, {}
    try:
        df = pd.read_csv(uploaded_file)
        if 'Numero' not in df.columns or 'Atraso' not in df.columns or 'Frecuencia' not in df.columns:
            st.error("El archivo debe contener las columnas 'Numero', 'Atraso' y 'Frecuencia'.")
            return None, {}, {}, {}, [], {}, 0, {}

        df['Numero'] = pd.to_numeric(df['Numero'], errors='coerce')
        df['Atraso'] = pd.to_numeric(df['Atraso'], errors='coerce')
        df['Frecuencia'] = pd.to_numeric(df['Frecuencia'], errors='coerce')
        df.dropna(subset=['Numero', 'Atraso', 'Frecuencia'], inplace=True)
        df['Numero'], df['Atraso'], df['Frecuencia'] = df['Numero'].astype(int).astype(str), df['Atraso'].astype(int), df['Frecuencia'].astype(int)

        st.success("Archivo de datos cargado exitosamente.")
        numero_a_atraso = dict(zip(df['Numero'], df['Atraso']))
        numero_a_frecuencia = dict(zip(df['Numero'], df['Frecuencia']))
        atrasos_disponibles_int = sorted(df['Atraso'].unique())
        numeros_validos = list(numero_a_atraso.keys())
        distribucion_probabilidad = {num: 1.0/len(numeros_validos) for num in numeros_validos} if numeros_validos else {}
        atraso_counts = df['Atraso'].astype(str).value_counts().to_dict()
        total_atraso_dataset = df['Atraso'].sum()
        atraso_stats = {"min": df['Atraso'].min(), "max": df['Atraso'].max(), "p25": df['Atraso'].quantile(0.25), "p75": df['Atraso'].quantile(0.75)}

        return df, numero_a_atraso, numero_a_frecuencia, distribucion_probabilidad, atrasos_disponibles_int, atraso_counts, total_atraso_dataset, atraso_stats
    except Exception as e:
        st.error(f"Error al procesar el archivo de datos: {e}")
        return None, {}, {}, {}, [], {}, 0, {}

@st.cache_data
def load_historical_combinations(uploaded_file):
    if uploaded_file is None: return []
    try:
        df_hist = pd.read_csv(uploaded_file, header=None)
        historical_sets = [set(pd.to_numeric(row, errors='coerce').dropna().astype(int)) for _, row in df_hist.iterrows()]
        historical_sets = [s for s in historical_sets if len(s) >= 6]
        if historical_sets: st.success(f"Archivo de historial cargado: {len(historical_sets)} combinaciones.")
        else: st.warning("El archivo de historial no contenía combinaciones válidas.")
        return historical_sets
    except Exception as e:
        st.error(f"Error al procesar el archivo de historial: {e}")
        return []

# --- Funciones de Análisis Histórico ---
@st.cache_data
def analyze_historical_special_calc(historical_sets, total_atraso_dataset, numero_a_atraso):
    if not historical_sets or total_atraso_dataset is None: return None
    values = [total_atraso_dataset + 40 - sum(numero_a_atraso.get(str(num), 0) for num in s) for s in historical_sets]
    if not values: return None
    return {"min": int(np.min(values)), "max": int(np.max(values)), "mean": int(np.mean(values)), "std": int(np.std(values))}

@st.cache_data
def analyze_historical_frequency_cv(historical_sets, numero_a_frecuencia):
    if not historical_sets or not numero_a_frecuencia: return None
    cv_values = []
    for s in historical_sets:
        freqs = [numero_a_frecuencia.get(str(num), 0) for num in s]
        if np.mean(freqs) > 0:
            cv = np.std(freqs) / np.mean(freqs)
            cv_values.append(cv)
    if not cv_values: return None
    return {"min": np.min(cv_values), "max": np.max(cv_values), "mean": np.mean(cv_values), "std": np.std(cv_values)}

@st.cache_data
def analyze_historical_composition(historical_sets, numero_a_atraso, composicion_ranges):
    if not historical_sets: return None
    def get_category(atraso, ranges):
        if ranges['caliente'][0] <= atraso <= ranges['caliente'][1]: return 'caliente'
        elif ranges['tibio'][0] <= atraso <= ranges['tibio'][1]: return 'tibio'
        elif ranges['frio'][0] <= atraso <= ranges['frio'][1]: return 'frio'
        elif atraso >= ranges['congelado'][0]: return 'congelado'
        return 'otro'
    counts = Counter(tuple(Counter(get_category(numero_a_atraso.get(str(num), -1), composicion_ranges) for num in s).get(cat, 0) for cat in ['caliente', 'tibio', 'frio', 'congelado']) for s in historical_sets)
    return counts if counts else None

# --- Motores de Generación y Filtrado ---
def generar_combinaciones_con_restricciones(dist_prob, num_a_atraso, num_a_freq, restr_atraso, n_sel, n_comb, hist_combs, total_atraso, special_range, freq_cv_range):
    valores = list(dist_prob.keys())
    combinaciones = []
    intentos = 0
    max_intentos = n_comb * 350
    while len(combinaciones) < n_comb and intentos < max_intentos:
        intentos += 1
        seleccionados = random.sample(valores, n_sel)
        
        freqs = [num_a_freq.get(val, 0) for val in seleccionados]
        mean_freq = np.mean(freqs)
        if mean_freq == 0: continue
        cv_freq = np.std(freqs) / mean_freq
        if not (freq_cv_range[0] <= cv_freq <= freq_cv_range[1]): continue

        if any(Counter(num_a_atraso.get(n, -1) for n in seleccionados)[int(a)] > l for a, l in restr_atraso.items()): continue
        suma_atrasos = sum(num_a_atraso.get(val, 0) for val in seleccionados)
        valor_especial = total_atraso + 40 - suma_atrasos
        if not (special_range[0] <= valor_especial <= special_range[1]): continue
        if hist_combs and any(len(set(int(n) for n in seleccionados).intersection(h)) > 2 for h in hist_combs): continue
        
        seleccionados.sort(key=int)
        combinaciones.append(tuple(seleccionados))
        
    conteo = Counter(combinaciones)
    return sorted({c: (f, np.prod([dist_prob.get(v, 0) for v in c])) for c, f in conteo.items()}.items(), key=lambda x: (-x[1][1], -x[1][0]))

def procesar_combinaciones(dist_prob, num_a_atraso, num_a_freq, restr_atraso, n_sel, n_comb, n_ejec, hist_combs, total_atraso, special_range, freq_cv_range):
    args = (dist_prob, num_a_atraso, num_a_freq, restr_atraso, n_sel, n_comb, hist_combs, total_atraso, special_range, freq_cv_range)
    with ProcessPoolExecutor() as executor:
        return [future.result() for future in as_completed([executor.submit(generar_combinaciones_con_restricciones, *args) for _ in range(n_ejec)])]

def filtrar_por_composicion(combinaciones, numero_a_atraso, composicion_rules):
    def get_category(atraso, ranges):
        if ranges['caliente'][0] <= atraso <= ranges['caliente'][1]: return 'caliente'
        elif ranges['tibio'][0] <= atraso <= ranges['tibio'][1]: return 'tibio'
        elif ranges['frio'][0] <= atraso <= ranges['frio'][1]: return 'frio'
        elif atraso >= ranges['congelado'][0]: return 'congelado'
        return 'otro'
    return [c for c in combinaciones if all(Counter(get_category(numero_a_atraso.get(str(n),-1), composicion_rules['ranges']) for n in c).get(cat,0)==cnt for cat,cnt in composicion_rules['counts'].items())]

def evaluar_individuo_deap(individuo, dist_prob, num_a_atraso, num_a_freq, restr_atraso, n_sel, hist_combs, total_atraso, special_range, freq_cv_range):
    if len(individuo) != n_sel or len(set(individuo)) != n_sel: return (0,)
    
    freqs = [num_a_freq.get(val, 0) for val in individuo]
    mean_freq = np.mean(freqs)
    if mean_freq == 0: return (0,)
    cv_freq = np.std(freqs) / mean_freq
    if not (freq_cv_range[0] <= cv_freq <= freq_cv_range[1]): return (0,)
        
    if any(Counter(num_a_atraso.get(val) for val in individuo if num_a_atraso.get(val) is not None).get(a, 0) > l for a, l in restr_atraso.items()): return (0,)
    if hist_combs and any(len(set(int(n) for n in individuo).intersection(h)) > 2 for h in hist_combs): return (0,)
    suma_atrasos = sum(num_a_atraso.get(val, 0) for val in individuo)
    valor_especial = total_atraso + 40 - suma_atrasos
    if not (special_range[0] <= valor_especial <= special_range[1]): return (0,)
    
    return (np.prod([dist_prob.get(val, 0) for val in individuo]),)

def ejecutar_algoritmo_genetico(n_gen, n_pob, cxpb, mutpb, dist_prob, num_a_atraso, num_a_freq, restr_atraso, hist_combs, total_atraso, special_range, freq_cv_range, n_sel=6):
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, list(dist_prob.keys()), n_sel)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluar_individuo_deap, dist_prob=dist_prob, num_a_atraso=num_a_atraso, num_a_freq=num_a_freq, restr_atraso=restr_atraso, n_sel=n_sel, hist_combs=hist_combs, total_atraso=total_atraso, special_range=special_range, freq_cv_range=freq_cv_range)
    toolbox.register("mate", tools.cxTwoPoint); toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1); toolbox.register("select", tools.selTournament, tournsize=3)
    pop = toolbox.population(n=n_pob)
    algorithms.eaSimple(pop, toolbox, cxpb, mutpb, n_gen, verbose=False)
    best = tools.selBest(pop, k=1)[0] if pop else None
    if best:
        best = sorted(list(set(best)), key=int)
        if len(best) != n_sel: return None, 0.0, "AG no mantuvo individuos válidos."
        return best, toolbox.evaluate(best)[0], None
    return None, 0.0, "Población final vacía."

# ----------------------- Interfaz Gráfica de Streamlit -----------------------

st.set_page_config(layout="wide", page_title="Generador de Combinaciones Homeostático")
st.title("Modelo Homeostático de Generación de Combinaciones")

if 'suggested_composition' not in st.session_state:
    st.session_state.suggested_composition = None

st.header("1. Cargar Archivos de Datos")
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Sube CSV ('Numero', 'Atraso', 'Frecuencia')", type="csv", key="data_uploader")
    df, num_a_atraso, num_a_freq, dist_prob, atrasos_disp, atraso_counts, total_atraso, atraso_stats = load_data_and_counts(uploaded_file)
with col2:
    hist_file = st.file_uploader("Sube CSV con Historial de Combinaciones", type="csv", key="history_uploader")
    historical_combinations_set = load_historical_combinations(hist_file)

n_selecciones = 6
if df is not None:
     st.info(f"**Suma total de 'Atraso' en el dataset:** {total_atraso}")

st.header("2. Configurar Filtros de Homeostasis (Etapa 1)")
restricciones_finales, composicion_rules = {}, {}
special_calc_range, freq_cv_range = (0, 99999), (0.0, 999.9)

if df is not None:
    if historical_combinations_set:
        col_freq, col_spec = st.columns(2)
        with col_freq:
            with st.expander("Filtro por CV de Frecuencia (Largo Plazo)", expanded=True):
                stats_freq_cv = analyze_historical_frequency_cv(historical_combinations_set, num_a_freq)
                if stats_freq_cv:
                    st.info(f"Historial: CV de Frecuencia varía de **{stats_freq_cv['min']:.2f}** a **{stats_freq_cv['max']:.2f}**, con un promedio de **{stats_freq_cv['mean']:.2f}**.")
                    
                    # CORRECCIÓN: Sujetar los valores por defecto para que estén dentro de los límites del slider.
                    slider_min_cv = 0.0
                    slider_max_cv = 2.0
                    default_start_cv = stats_freq_cv['mean'] - stats_freq_cv['std']
                    default_end_cv = stats_freq_cv['mean'] + stats_freq_cv['std']
                    
                    clamped_start_cv = max(slider_min_cv, default_start_cv)
                    clamped_end_cv = min(slider_max_cv, default_end_cv)
                    
                    freq_cv_range = st.slider("Rango deseado para CV de Frecuencia:", slider_min_cv, slider_max_cv, (clamped_start_cv, clamped_end_cv), format="%.2f", key="freq_cv_slider")

        with col_spec:
            with st.expander("Filtro de 'Cálculo Especial' (Corto Plazo)", expanded=True):
                stats_special = analyze_historical_special_calc(historical_combinations_set, total_atraso, num_a_atraso)
                if stats_special:
                    st.info(f"Historial: 'Cálculo Especial' varía de **{stats_special['min']}** a **{stats_special['max']}**.")
                    
                    # CORRECCIÓN: Sujetar los valores por defecto para que estén dentro de los límites del slider.
                    slider_min_special = float(stats_special['min'] - 50)
                    slider_max_special = float(stats_special['max'] + 50)
                    default_start_special = float(stats_special['mean'] - stats_special['std'])
                    default_end_special = float(stats_special['mean'] + stats_special['std'])

                    clamped_start_special = max(slider_min_special, default_start_special)
                    clamped_end_special = min(slider_max_special, default_end_special)

                    special_calc_range = st.slider("Rango deseado:", min_value=slider_min_special, max_value=slider_max_special, value=(clamped_start_special, clamped_end_special), key="special_slider")

    with st.expander("Filtro de Atrasos Individuales y Composición (Etapa 1 y 2)"):
        st.subheader("Filtro de Atrasos Individuales (Etapa 1)")
        selected_atrasos = st.multiselect("Selecciona 'Atraso' a restringir:", options=[str(a) for a in atrasos_disp], default=[str(a) for a in atrasos_disp])
        cols_ui_atraso = st.columns(4)
        for i, atraso_str in enumerate(selected_atrasos):
            with cols_ui_atraso[i % 4]:
                limit = st.number_input(f"Max Atraso '{atraso_str}':", 0, n_selecciones, atraso_counts.get(atraso_str, 0), key=f"res_{atraso_str}")
                restricciones_finales[atraso_str] = limit
        
        st.subheader("Filtro Estratégico de Composición (Etapa 2)")
        max_atraso = atraso_stats.get("max", 100)
        c1, c2 = st.columns(2)
        with c1:
            range_caliente = st.slider("Rango 'Caliente'", 0, max_atraso, (0, int(atraso_stats.get("p25", 5))), key="r_hot")
            range_frio = st.slider("Rango 'Frío'", 0, max_atraso, (int(atraso_stats.get("p75", 15)), max_atraso - 1), key="r_cold")
        with c2:
            range_tibio = st.slider("Rango 'Tibio'", 0, max_atraso, (range_caliente[1] + 1, range_frio[0] -1), key="r_warm")
            min_congelado = st.number_input("Atraso mínimo 'Congelado'", value=max_atraso, key="r_icy")
        current_ranges = {'caliente': range_caliente, 'tibio': range_tibio, 'frio': range_frio, 'congelado': (min_congelado, 9999)}
        
        if historical_combinations_set:
            st.write("**Análisis y Recomendación de Composición Histórica**")
            comp_analysis = analyze_historical_composition(historical_combinations_set, num_a_atraso, current_ranges)
            if comp_analysis:
                most_common = comp_analysis.most_common(5)
                most_common_comp, _ = most_common[0]
                st.success(f"**Composición más frecuente:** {most_common_comp[0]} Calientes, {most_common_comp[1]} Tibios, {most_common_comp[2]} Fríos, {most_common_comp[3]} Congelados")
                if st.button("Aplicar Composición Sugerida"):
                    st.session_state.suggested_composition = most_common_comp; st.rerun()
        
        suggested = st.session_state.suggested_composition
        c3, c4, c5, c6 = st.columns(4)
        count_caliente = c3.number_input("Nº Calientes", 0, n_selecciones, suggested[0] if suggested else 2, key="c_hot")
        count_tibio = c4.number_input("Nº Tibios", 0, n_selecciones, suggested[1] if suggested else 2, key="c_warm")
        count_frio = c5.number_input("Nº Fríos", 0, n_selecciones, suggested[2] if suggested else 2, key="c_cold")
        count_congelado = c6.number_input("Nº Congelados", 0, n_selecciones, suggested[3] if suggested else 0, key="c_icy")

        total_count_composition = count_caliente + count_tibio + count_frio + count_congelado
        if total_count_composition == n_selecciones:
            st.success("La composición es válida.")
            composicion_rules = {'ranges': current_ranges, 'counts': {'caliente': count_caliente, 'tibio': count_tibio, 'frio': count_frio, 'congelado': count_congelado}}
        else: st.warning(f"La suma ({total_count_composition}) debe ser igual a {n_selecciones}.")

    with st.expander("Configurar Parámetros de los Algoritmos"):
        col_ga, col_sim = st.columns(2)
        with col_ga: st.subheader("Algoritmo Genético"); ga_ngen=st.slider("Generaciones",10,1000,200); ga_npob=st.slider("Población",100,5000,1000); ga_cxpb=st.slider("Cruce",0.0,1.0,0.7); ga_mutpb=st.slider("Mutación",0.0,1.0,0.2)
        with col_sim: st.subheader("Simulación en Cascada"); sim_n_comb=st.number_input("Combinaciones/Ejec.",1000,value=50000); sim_n_ejec=st.number_input("Ejecuciones",1,value=8)
else:
    st.info("Carga un archivo de Datos ('Numero', 'Atraso', 'Frecuencia') para empezar.")

# ... (código anterior) ...

st.header("3. Ejecutar Algoritmos")
if df is not None:
    run_col1, run_col2 = st.columns(2)
    with run_col1:
        if st.button("Ejecutar Algoritmo Genético"):
            with st.spinner("Buscando la mejor combinación..."):
                # CORRECCIÓN: Se cambió 'mutpb' a 'ga_mutpb' para que coincida con la definición del slider.
                mejor_ind, _, err_msg = ejecutar_algoritmo_genetico(
                    ga_ngen, ga_npob, ga_cxpb, ga_mutpb, 
                    dist_prob, num_a_atraso, num_a_freq, 
                    restricciones_finales, historical_combinations_set, 
                    total_atraso, special_calc_range, freq_cv_range, n_selecciones
                )
            if err_msg: st.error(err_msg)
            elif mejor_ind:
                st.subheader("Mejor Combinación (GA)")
                st.success(f"**Combinación: {' - '.join(map(str, mejor_ind))}**")
                freqs = [num_a_freq.get(v, 0) for v in mejor_ind]
                st.write(f"**CV Frecuencia:** {np.std(freqs) / np.mean(freqs):.2f}")
                st.write(f"**Cálculo Especial:** {total_atraso + 40 - sum(num_a_atraso.get(v, 0) for v in mejor_ind)}")
            else: st.warning("El GA no encontró una combinación válida.")



    with run_col2:
        if st.button("Ejecutar Simulación en Cascada"):
            with st.spinner("Etapa 1: Generando combinaciones en paralelo..."):
                start_time = time.time()
                resultados = procesar_combinaciones(dist_prob, num_a_atraso, num_a_freq, restricciones_finales, n_selecciones, sim_n_comb, sim_n_ejec, historical_combinations_set, total_atraso, special_calc_range, freq_cv_range)
                st.info(f"Etapa 1 completada en {time.time() - start_time:.2f} s. Se encontraron {sum(len(r) for r in resultados)} combinaciones válidas.")
            
            todas_unicas = list(set(tuple(int(n) for n in c) for res in resultados for c, _ in res))
            st.info(f"Se generaron **{len(todas_unicas)}** combinaciones únicas en la Etapa 1.")

            combinaciones_refinadas = []
            if total_count_composition == n_selecciones:
                with st.spinner("Etapa 2: Aplicando filtro de composición..."):
                    combinaciones_refinadas = filtrar_por_composicion(todas_unicas, num_a_atraso, composicion_rules)
                st.success(f"Etapa 2 completada. Se encontraron **{len(combinaciones_refinadas)}** combinaciones que cumplen el perfil.")
            
            st.subheader(f"Resultados del Filtro (Etapa 2) - {len(combinaciones_refinadas)} combinaciones")
            if combinaciones_refinadas:
                data = []
                for c in combinaciones_refinadas:
                    freqs = [num_a_freq.get(str(v),0) for v in c]
                    data.append({"Combinación": " - ".join(map(str, sorted(c))), "CV Frecuencia": np.std(freqs)/np.mean(freqs), "Cálculo Especial": total_atraso + 40 - sum(num_a_atraso.get(str(v),0) for v in c)})
                df_results = pd.DataFrame(data)
                df_results['CV Frecuencia'] = df_results['CV Frecuencia'].map('{:,.2f}'.format)
                st.dataframe(df_results.reset_index(drop=True))
            else: st.info("Ninguna combinación superó el filtro de la Etapa 2.")

st.sidebar.header("Guía del Modelo")
st.sidebar.markdown("Este modelo se basa en el **principio de homeostasis**: un sistema aleatorio tiende al equilibrio.")
st.sidebar.markdown("**Filtros de Homeostasis (Etapa 1):**")
st.sidebar.markdown("- **CV de Frecuencia (Largo Plazo):** Busca combinaciones con miembros de popularidad similar, reflejando el equilibrio del sistema.")
st.sidebar.markdown("- **Cálculo Especial (Corto Plazo):** Busca combinaciones que mantengan la suma de atrasos del sistema en un rango estable.")
st.sidebar.markdown("**Filtro Estratégico (Etapa 2):**")
st.sidebar.markdown("- **Composición:** Define la 'personalidad' de la combinación (Calientes, Fríos, etc.). La app **recomienda la estrategia más común** del historial.")
