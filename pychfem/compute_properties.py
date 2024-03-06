from pychfem.wrapper import run

def computer_permeability(ws, nf):
    
    print("Running chfem")
    run(nf, ws)
