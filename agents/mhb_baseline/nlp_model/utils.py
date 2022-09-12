def parse_logs(logs, history_len=None):
    if history_len:
        logs = logs[-history_len:]
    input_line = ''
    for item in logs:
        input_line += f'<Architect> {item[0]} <Builder> {item[1]} <sep1> '
    return input_line

colours = {'red' : 1, 'orange' : 2, 'yellow' : 3, 'green' : 4, 'blue' : 5, 'purple' : 6}

x_orientation = ['right', 'left']
y_orientation = ['lower', 'upper']
z_orientation = ['before', 'after']


def logging(log_file, log):
    if not log_file:
        print(log)
    else:
        log_file.writelines(log+'\n')

def parse_coords(action, x_0=0, y_0=0, z_0=0):
    x_left_find = action.find(x_orientation[0])
    x_right_find = action.find(x_orientation[1])
    if x_right_find != -1:
        x_pos = x_0 + int(action[x_right_find - 3:x_right_find - 1])
    elif x_left_find != -1:
        x_pos = x_0 - int(action[x_left_find - 3:x_left_find - 1])
    else:
        x_pos = x_0
    
    y_left_find = action.find(y_orientation[0])
    y_right_find = action.find(y_orientation[1])
    if y_right_find != -1:
        y_pos = y_0 + int(action[y_right_find - 2])
    elif y_left_find != -1:
        y_pos = y_0 - int(action[y_left_find - 2])
    else:
        y_pos = y_0
    
    z_left_find = action.find(z_orientation[0])
    z_right_find = action.find(z_orientation[1])
    if z_right_find != -1:
        z_pos = z_0 + int(action[z_right_find - 3:z_right_find - 1])
    elif z_left_find != -1:
        z_pos = z_0 - int(action[z_left_find - 3:z_left_find - 1])
    else:
        z_pos = z_0
        
    return x_pos, y_pos, z_pos
        

def update_state_from_action(args, state, action, init_block, log_file=None):
    x_0, y_0, z_0 = init_block
    action = action.strip()
    command = action.split()[0]
    ok = True
   #print(action)
    if not command in ['pick', 'put']:
        if args.verbose > 1:
            logging(log_file, f'action: {action}, Invalid command {command}')
        ok = False
        return state, ok
    
    if action.split()[1] == 'initial':
        colour = action.split()[2]
        if not colour in colours.keys():
            if args.verbose > 1:
                logging(log_file, f'action: {action}, Invalid colour {colour}')
            ok = False
            return state, ok
        
        if command == 'put':
            if state[init_block] == 0:
                state[init_block] = colours[colour]
            else:
                if args.verbose > 1:
                    logging(log_file, f'action: {action}, Wrong action, position occupied')
                ok = False
                return state, ok
        else:
            if state[init_block] == colours[colour]:
                state[init_block] = 0
            else:
                if args.verbose > 1:
                    logging(log_file, 
                            f'action: {action} Wrong action, colours dont match: {state[init_block]}, {colours[colour]}')
                ok = False
                return state, ok
    else:
        colour = action.split()[1]
        if not colour in colours.keys():
            if args.verbose > 1:
                logging(log_file, f'action: {action}, Invalid colour {colour}')
            ok = False
            return state, ok
            
        x, y, z = parse_coords(action, x_0, y_0, z_0)
        if (x not in range(0, 11)) or (y not in range(0, 9)) or (z not in range(0, 11)):
            if args.verbose > 1:
                logging(log_file, f'action: {action}, Invalid coordinates {x}, {y}, {z}')
            ok = False
            return state, ok
        
        if command == 'put':
            if state[x, y, z] == 0:
                state[x, y, z] = colours[colour]
            else:
                if args.verbose > 1:
                    logging(log_file, f'action: {action}, Wrong action, position occupied')
                ok = False
                return state, ok
            
        else:
            if state[x, y, z] == colours[colour]:
                state[x, y, z] = 0
            else:
                if args.verbose > 1:
                    logging(log_file, 
                            f'action: {action}, Wrong action, colours dont match: {state[x, y, z]}, {colours[colour]}')
                ok = False
                return state, ok
                
    return state, ok
