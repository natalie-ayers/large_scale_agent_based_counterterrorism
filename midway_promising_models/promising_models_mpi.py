from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time
import model_mpi_midway
import pickle
import pandas as pd
import seaborn as sns
from math import ceil


def sim_counterterrorism_models():
    """
    Run counterterrorism model distributed across nodes using MPI,
        use parameters stored in 'promising_params.p'. Store the results
        and create visualizations for some pre-specified model choices.
    """

    # Get rank of process and overall size of communicator:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Start time:
    t0 = time.time()

    if rank == 0:
        pickled_params = pickle.load(open("promising_params.p","rb"))

        # calculate number of different paramaterized models
        n_runs = len(pickled_params)
        print('total parameter combinations to consider:',n_runs)

        N = ceil(n_runs / size)
        # create list of lists to scatter to each node
        params_resp = [pickled_params[i:i+N] for i in range(0, n_runs, N)]
        # keep ability to run smaller subset for testing
        #params_resp = [params[40:43] for params in params_resp]
    else:
        params_resp = None

    # scatter assigned parameters to each node
    params_resp = comm.scatter(params_resp, root=0)
    print('params_resp for rank',rank,':',params_resp)

    # run the model with the assigned parameters
    results_dicts = []
    for params in params_resp:
        return_dict = {'params':params}
        grid_size = {200:(2,3),300:(3,3),400:(3,4),500:(4,4),600:(4,5),\
                  800:(5,5)}

        prob_violence = params['prob_violence']
        govt_policy = params['govt_policy']
        reactive_lvl = params['reactive_lvl']
        discontent = params['discontent']
        starting_population = params['starting_population']
        steps = params['steps']

        height, width = grid_size[starting_population]
        model = model_mpi_midway.CounterterrorismModel(N=starting_population,height=height,\
                                                width=width,prob_violence=prob_violence,\
                                                policy=govt_policy,reactive_lvl=reactive_lvl,\
                                                discontent=discontent)
        for i in range(steps):
            model.step()

        # obtain data/results from model's run
        model_df = model.datacollector.get_model_vars_dataframe()
        agents_df = model.datacollector.get_agent_vars_dataframe()
        deaths_df = model.datacollector.get_table_dataframe('Deaths')

        # get values from model dataframe
        return_dict['final_pop'] = int(model_df[model_df.index == steps-1]\
            ['num_agents'].values[0])
        return_dict['total_num_attacks'] = model_df[model_df.index == steps-1]\
            ['num_attacks'].values[0]

        # process agents dataframe
        agents_df = agents_df.reset_index()
        agents_df['step_cat'] = agents_df.Step.astype('str')
        agent_stati_gb = agents_df.groupby(by=['step_cat','status'])
        agent_stati = agent_stati_gb['AgentID'].nunique().reset_index()
        agent_stati = agent_stati.rename(columns={'AgentID':'num_agents'})
        agent_stati['step'] = agent_stati.step_cat.astype('int')

        # get Palestinian actors dataframe
        palestinian_stati = agent_stati[agent_stati.status.isin(['anti-violence','combatant','neutral','sympathetic'])]
        return_dict['palestinian_stati'] = palestinian_stati

        # get dominant sentiments and proportion of sentiments at end 
        dominant_sentiments = np.zeros((model.grid.width, model.grid.height))
        dominant_sentiments = dominant_sentiments.astype(str)
        dominant_sentiments[dominant_sentiments=='0.0'] = 'none'
        percent_dominant_sentiments = np.zeros((model.grid.width, model.grid.height))

        # calculate dominant sentiments for each grid cell in the model
        for cell in model.grid.coord_iter():
            cell_content, x, y = cell
            status_dict = {'anti-violence':0,'NONE':0,'neutral':0,'sympathetic':0,\
                'combatant':0,'TARG-CONC':0,'TARG-REPR':0,'INDISC-CONC':0,'INDISC-REPR':0}
            for agent in cell_content:
                status_dict[agent.status] += 1
                #print(agent.status) 
            dominant_sentiment = max(status_dict, key = lambda x: status_dict[x])
            if len(cell_content) > 0:
                perc_dominant_sentiment = max(status_dict.values())/len(cell_content)
            else:
                perc_dominant_sentiment = 0

            dominant_sentiments[x][y] = dominant_sentiment 
            percent_dominant_sentiments[x][y] = perc_dominant_sentiment  

        return_dict['dominant_sentiments'] = dominant_sentiments
        return_dict['percent_dominant_sentiments'] = percent_dominant_sentiments

        # get Israeli government actions dataframes
        govt_status = agent_stati[agent_stati.status.isin(['NONE','INDISC-REPR',\
                                                            'INDISC-CONC','TARG-CONC',\
                                                            'TARG-REPR'])]
        govt_status.drop('num_agents', axis=1,inplace=True)
        govt_status = govt_status.sort_values('step',ascending=True)
        govt_status_cum = govt_status.groupby(['status']).cumcount()
        govt_status_cum = govt_status_cum.rename('cumulative_actions')
        govt_status_cum = govt_status.join(govt_status_cum)

        return_dict['govt_status'] = govt_status
        return_dict['govt_status_cum'] = govt_status_cum 
        return_dict['num_targ_conc'] = govt_status_cum[govt_status_cum.status == \
                                                        'TARG-CONC']['cumulative_actions'].max()
        return_dict['num_indisc_conc'] = govt_status_cum[govt_status_cum.status == \
                                                        'INDISC-CONC']['cumulative_actions'].max()
        return_dict['num_targ_repr'] = govt_status_cum[govt_status_cum.status == \
                                                        'TARG-REPR']['cumulative_actions'].max()
        return_dict['num_indisc_repr'] = govt_status_cum[govt_status_cum.status == \
                                                        'INDISC-REPR']['cumulative_actions'].max()
        
        # add raw dataframes to dicts
        return_dict['deaths_df'] = deaths_df
        return_dict['model_df'] = model_df

        results_dicts.append(return_dict)

        model_time = time.time() - t0
        print('model with params',params,'took',model_time,'to run on rank',rank)

    # gather final results
    final_results_lol = comm.gather(results_dicts, root=0)

    # process final results on root 0 node
    # create visualizations and pickle full dataframes
    if rank == 0:
        final_results = []
        for lol in final_results_lol:
            final_results.extend(lol)

        # find model with highest number of attacks
        # create visualizations of this model's results
        attacks_idx, highest_attacks = find_dict(final_results, 'total_num_attacks')
        max_attacks = final_results[attacks_idx]
        print('max attacks occurred with params:',max_attacks['params'])
        create_visualizations(max_attacks, 'max_attacks', highest_attacks)

        # find model with highest final population
        # create visualizations of this model's results
        pop_idx, final_pop = find_dict(final_results, 'final_pop')
        highest_pop = final_results[pop_idx]
        print('highest pop occurred with:',highest_pop['params'])
        create_visualizations(highest_pop, 'highest_pop', final_pop)

        # pickle and store final dataframes
        final_statistics = create_final_output(final_results)
        print('final results from all runs:',final_statistics)

        total_time = time.time() - t0
        print('overall runtime for',n_runs,'parameter combinations on',size,'nodes\
        was',total_time)

    return


def find_dict(dict_list, key):
    """
    Find the model with the highest value of the provided key
    Inputs:
        dict_list (list of model dictionary results)
        key (str): model value to evaluate
    Return:
        idx_highest (int): index location of desired model
        highest (int): highest value 
    """
    highest = 0
    idx_highest = 0
    for idx, dict in enumerate(dict_list):
        dict_val = dict[key]
        if dict_val > highest:
            highest = dict_val
            idx_highest = idx
    return idx_highest, highest


def create_final_output(dict_list):
    """
    Create final, full dataframes and results dictionaries to save for 
    any future required study
    Inputs:
        dict_list (list of model dictionary results)
    """
    # create empty dataframes for concatenating
    palestinian_stati_df = pd.DataFrame(columns=['step_cat','status',\
                                                'num_agents','step','params'])
    govt_status_df = pd.DataFrame(columns=['step_cat','status','step','params'])
    govt_status_cum_df = pd.DataFrame(columns=['step_cat','status',\
                                                'step','cumulative_actions','params'])
    full_deaths_df = pd.DataFrame(columns=['step','deaths','params'])
    full_models_df = pd.DataFrame(columns=['num_agents','num_attacks','params'])
    final_vals = []

    # create full dataframes 
    for idx, model_dict in enumerate(dict_list):
        param_vals = model_dict['params']
        #print('considering run with params',param_vals)
        palestinian_df = model_dict['palestinian_stati']
        palestinian_df = palestinian_df.assign(params = str(param_vals))
        govt_status = model_dict['govt_status']
        govt_status = govt_status.assign(params = str(param_vals))
        govt_status_cum = model_dict['govt_status_cum']
        govt_status_cum = govt_status_cum.assign(params = str(param_vals))
        deaths_df = model_dict['deaths_df']
        deaths_df = deaths_df.assign(params = str(param_vals))
        model_df = model_dict['model_df']
        model_df = model_df.assign(params = str(param_vals))

        palestinian_stati_df = palestinian_stati_df.append(palestinian_df, \
                                                            ignore_index=True)
        govt_status_df = govt_status_df.append(govt_status, ignore_index=True)
        govt_status_cum_df = govt_status_cum_df.append(govt_status_cum, \
                                                        ignore_index=True)
        full_deaths_df = full_deaths_df.append(deaths_df, ignore_index=True)
        full_models_df = full_models_df.append(model_df, ignore_index=True)

        final_vals.append({'params':param_vals,\
                                'final_pop':model_dict['final_pop'],\
                                'total_num_attacks':model_dict['total_num_attacks'],\
                                'dominant_sentiments':model_dict['dominant_sentiments'].tolist(),\
                                'percent_dominant_sentiments':\
                                    model_dict['percent_dominant_sentiments'].tolist(),\
                                'num_targ_conc':model_dict['num_targ_conc'],\
                                'num_indisc_conc':model_dict['num_indisc_conc'],\
                                'num_targ_repr':model_dict['num_targ_repr'],\
                                'num_indisc_repr':model_dict['num_indisc_repr']
                                })

    palestinian_stati_df.to_pickle('palestinian_stati_df.pkl')
    govt_status_df.to_pickle('govt_status_df.pkl')
    govt_status_cum_df.to_pickle('govt_status_cum_df.pkl')
    full_deaths_df.to_pickle('full_deaths_df.pkl')
    full_models_df.to_pickle('full_models_df.pkl')

    return final_vals


def create_visualizations(result_dict, file_suffix, identifier):
    """
    Create useful visualizations and save for the provided results
    of a model's run (one of the result_dicts objects)
    Inputs:
        result_dict: a single object from the results_dict list
        file_suffix: any desired text to append to the name of the
        visualizations for this run
        identifier: a summary statistic from the results to append to
        image name
    """

    result_dict['model_df'].plot()
    plt.xlabel('Model Steps')
    plt.ylabel('Total Count (Agents or Attacks)')
    plt.title('Number of Agents and Cumulative Attacks')
    plt.savefig('attacks-and-agents-%s-%d.png' % (file_suffix, identifier))

    sns.set(style='darkgrid')
    plt.figure(figsize=(20,10))
    params = result_dict['params']
    steps = params['steps']
    max_steps = result_dict['palestinian_stati'].step.max()
    #print('steps used to create barplot',max_steps)

    sns.barplot(x='step',y='num_agents',hue='status',\
                data=result_dict['palestinian_stati']\
                [result_dict['palestinian_stati'].\
                    step.isin(range(1,max_steps,round(max_steps*0.05)))])
    plt.title('Progression of Palestinian Statuses')
    plt.savefig('palestinian_stati_bar-%s-%d.png' % (file_suffix, identifier))

    plt.figure(figsize=(20,10))
    vals_to_int = {j:i for i, j in enumerate(pd.unique(result_dict['dominant_sentiments'].ravel()))}
    n = len(vals_to_int)
    cmap = sns.color_palette('Spectral', n)
    dominant_sentiments_df = pd.DataFrame(result_dict['dominant_sentiments'])
    ax = sns.heatmap(dominant_sentiments_df.replace(vals_to_int),cmap=cmap)
    colorbar = ax.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
    colorbar.set_ticklabels(list(vals_to_int.keys()))
    ax.set_ylabel('grid height')
    ax.set_xlabel('grid width')
    ax.set_title('Dominant Final Sentiment Within Each Grid Cell')
    plt.savefig('dominant_sentiments-%s-%d.png' % (file_suffix, identifier))

    plt.figure(figsize=(20,10))
    ax = sns.heatmap(result_dict['percent_dominant_sentiments'], annot=True)
    ax.set_ylabel('grid height')
    ax.set_xlabel('grid width')
    ax.set_title('Percent of Population Expressing Dominant Sentiment')
    plt.savefig('perc_dominant_sentiments-%s-%d.png' % (file_suffix, identifier))

    plt.figure(figsize=(20,10))
    sns.lineplot(x='step',y='cumulative_actions',hue='status',\
                data=result_dict['govt_status_cum'])
    plt.title('Cumulative Israeli Government Actions')
    plt.savefig('govt_status_cumulative-%s-%d.png' % (file_suffix, identifier))
      

def main():
    sim_counterterrorism_models()

if __name__ == '__main__':
    main()