import numpy as np
import time
import model_lambda
import json


def lambda_handler(event, context):
    t0 = time.time()
    return_dict = {'params':event}
    prob_violence = float(event['prob_violence'])
    govt_policy = event['govt_policy']
    reactive_lvl = event['reactive_lvl']
    discontent = event['discontent']
    starting_population = int(event['starting_population'])
    steps = int(event['steps'])

    grid_size = {200:(2,3),300:(3,3),400:(3,4),500:(4,4),600:(4,5),\
                800:(5,5)}
    height, width = grid_size[starting_population]
    model = model_lambda.CounterterrorismModel(N=starting_population,height=height,\
                                            width=width,prob_violence=prob_violence,\
                                            policy=govt_policy,reactive_lvl=reactive_lvl,\
                                            discontent=discontent)
    for i in range(steps):
        model.step()

    model_df = model.datacollector.get_model_vars_dataframe()
    agents_df = model.datacollector.get_agent_vars_dataframe()
    deaths_df = model.datacollector.get_table_dataframe('Deaths')

    # get values from model dataframe
    return_dict['final_pop'] = int(model_df[model_df.index == steps-1]\
        ['num_agents'].values[0])
    return_dict['total_num_attacks'] = int(model_df[model_df.index == steps-1]\
        ['num_attacks'].values[0])

    # process agents dataframe
    agents_df = agents_df.reset_index()
    agents_df['step_cat'] = agents_df.Step.astype('str')
    agent_stati_gb = agents_df.groupby(by=['step_cat','status'])
    agent_stati = agent_stati_gb['AgentID'].nunique().reset_index()
    agent_stati = agent_stati.rename(columns={'AgentID':'num_agents'})
    agent_stati['step'] = agent_stati.step_cat.astype('int')

    # get Palestinian actors dataframe
    palestinian_stati = agent_stati[agent_stati.status.isin(['anti-violence','combatant','neutral','sympathetic'])]
    #return_dict['palestinian_stati'] = palestinian_stati

    # get dominant sentiments and proportion of sentiments at end 
    dominant_sentiments = np.zeros((model.grid.width, model.grid.height))
    dominant_sentiments = dominant_sentiments.astype(str)
    dominant_sentiments[dominant_sentiments=='0.0'] = 'none'
    percent_dominant_sentiments = np.zeros((model.grid.width, model.grid.height))

    for cell in model.grid.coord_iter():
        cell_content, x, y = cell
        status_dict = {'anti-violence':0,'NONE':0,'neutral':0,'sympathetic':0,\
            'combatant':0,'TARG-CONC':0,'TARG-REPR':0,'INDISC-CONC':0,'INDISC-REPR':0}
        for agent in cell_content:
            status_dict[agent.status] += 1
            #print(agent.status) 
        dominant_sentiment = max(status_dict)
        if len(cell_content) > 0:
            perc_dominant_sentiment = max(status_dict.values())/len(cell_content)
        else:
            perc_dominant_sentiment = 0

        dominant_sentiments[x][y] = dominant_sentiment 
        percent_dominant_sentiments[x][y] = perc_dominant_sentiment  

    return_dict['dominant_sentiments'] = dominant_sentiments.tolist()
    return_dict['percent_dominant_sentiments'] = percent_dominant_sentiments.tolist()

    # get Israeli government actions dataframes
    govt_status = agent_stati[agent_stati.status.isin(['NONE','INDISC-REPR',\
                                                        'INDISC-CONC','TARG-CONC',\
                                                        'TARG-REPR'])]
    govt_status.drop('num_agents', axis=1,inplace=True)
    govt_status = govt_status.sort_values('step',ascending=True)
    govt_status_cum = govt_status.groupby(['status']).cumcount()
    govt_status_cum = govt_status_cum.rename('cumulative_actions')
    govt_status_cum = govt_status.join(govt_status_cum)

    #return_dict['govt_status'] = govt_status
    #return_dict['govt_status_cum'] = govt_status_cum 
    return_dict['num_targ_conc'] = int(govt_status_cum[govt_status_cum.status == \
                                                    'TARG-CONC']['cumulative_actions'].max())
    return_dict['num_indisc_conc'] = int(govt_status_cum[govt_status_cum.status == \
                                                    'INDISC-CONC']['cumulative_actions'].max())
    return_dict['num_targ_repr'] = int(govt_status_cum[govt_status_cum.status == \
                                                    'TARG-REPR']['cumulative_actions'].max())
    return_dict['num_indisc_repr'] = int(govt_status_cum[govt_status_cum.status == \
                                                    'INDISC-REPR']['cumulative_actions'].max())
    
    # add raw dataframes to dicts
    #return_dict['deaths_df'] = deaths_df
    #return_dict['model_df'] = model_df

    
    model_time = time.time() - t0
    return_dict['time_elapsed'] = float(model_time)
    print('model with params',event,'took',model_time,\
            '\n \t resulting in',return_dict['total_num_attacks'],\
            'total attacks and a final population of',return_dict['final_pop'])

    return_dict_json = json.dumps(return_dict)
    
    return return_dict_json