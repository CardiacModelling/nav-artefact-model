from __future__ import print_function
import os
import sys
sys.path.append('../lib')
import numpy as np

import pints
import myokit
import myokit.pacing as pacing

###############################################################################
## Defining Model
###############################################################################

vhold = -80  # mV
# Default time
DT = 2.0e-1  # ms; maybe not rely on this...


#
# Time out handler
#
class Timeout(myokit.ProgressReporter):
    """
    A :class:`myokit.ProgressReporter` that halts the simulation after
    ``max_time`` seconds.
    """
    def __init__(self, max_time):
        self.max_time = float(max_time)
    def enter(self, msg=None):
        self.b = myokit.Benchmarker()
    def exit(self):
        pass
    def update(self, progress):
        return self.b.time() < self.max_time


#
# Create ForwardModel
#
class Model(pints.ForwardModel):
    parameters = [
        'ikr.g', 'ikr.p1', 'ikr.p2', 'ikr.p3', 'ikr.p4', 'ikr.p5', 'ikr.p6', 
        'ikr.p7', 'ikr.p8',  
        ]
    
    def __init__(self, model_file, protocol_def, temperature,
                 transform=None, useFilterCap=False, effEK=True,
                 readout='voltageclamp.Iin', max_evaluation_time=5):
        """
        # model_file: mmt model file for myokit
        # protocol_def: func take model as arg and return tuple
        #               (modified model, step)
        #               or
        #               str to a file name contains protocol time series.
        # temperature: temperature of the experiment to be set (in K).
        # transform: transform search space parameters to model parameters.
        # useFileterCap: bool, True apply filtering of capacitive spikes.
        # effEK: Fix to EK values for 37oC experiments
        # readout: myokit model component, set output for method simulate().
        # max_evaluation_time: maximum time (in second) allowed for one
        #                      simulate() call.
        """

        # maximum time allowed
        self.max_evaluation_time = max_evaluation_time
        
        # Load model
        model = myokit.load_model(model_file)
        # Set temperature
        try:
            model.get('phys.T').set_rhs(temperature)
            if temperature == 37.0 + 273.15 and effEK:
                print('Using effective EK for 37oC data')
                model.get('potassium.Ko').set_rhs(
                        110 * np.exp( -92.630662854828572 / \
                        (8.314472 * (273.15 + 37) / 9.64853415e4 * 1000))
                        )
        except KeyError:
            pass

        # Compute model EK
        const_R = 8.314472  # J/mol/K
        const_F = 9.64853415e4  # C/mol
        const_Ko = 4.0  # mM (my hERG experiments)
        const_Ki = 110.0  # mM
        RTF = const_R * temperature / const_F * 1e3  # J/C * 1e-3 == mV
        self._EK = RTF * np.log(const_Ko / const_Ki)

        # Set readout by simulate
        self._readout = readout

        # 1. Create pre-pacing protocol
        protocol = pacing.constant(vhold)
        # Create pre-pacing simulation
        self.simulation1 = myokit.Simulation(model, protocol)
        
        # 2. Create specified protocol
        self.useFilterCap = useFilterCap
        if type(protocol_def) is str:
            d = myokit.DataLog.load_csv(protocol_def).npview()
            self.simulation2 = myokit.Simulation(model)
            self.simulation2.set_fixed_form_protocol(
                d['time'] * 1e3,  # s -> ms
                d['voltage']  # mV
            )
            if self.useFilterCap:
                raise ValueError('Cannot use capacitance filtering with the'
                                 + ' given format of protocol_def')
        else:
            if self.useFilterCap:
                model, steps, fcap = protocol_def(model, self.useFilterCap)
                self.fcap = fcap
            else:
                model, steps = protocol_def(model)
            protocol = myokit.Protocol()
            for f, t in steps:
                protocol.add_step(f, t)
            # Create simulation for protocol
            self.simulation2 = myokit.Simulation(model, protocol)
        
        self.simulation2.set_tolerance(1e-8, 1e-10)
        self.simulation2.set_max_step_size(1e-1)

        self.transform = transform
        self.init_state = self.simulation1.state()

        self._fix_parameters = {}

    def set_parameters(self, parameters):
        # (Re-)Set the parameters to be inferred
        self.parameters = parameters

    def set_fix_parameters(self, parameters):
        # Set/update parameters to a fixed value
        self._fix_parameters = parameters

    def _set_fix_parameters(self, parameters):
        # Call to set parameters to a fixed value
        for p in parameters.keys():
            self.simulation1.set_constant(p, parameters[p])
            self.simulation2.set_constant(p, parameters[p])

    def n_parameters(self):
        # n_parameters() method for Pints
        return len(self.parameters)

    def cap_filter(self, times):
        if self.useFilterCap:
            return self.fcap(times)
        else:
            return None

    def simulate(self, parameters, times, extra_log=[]):
        # simulate() method for Pints

        # Update fix parameters
        self._set_fix_parameters(self._fix_parameters)

        # Update model parameters
        if self.transform is not None:
            parameters = self.transform(parameters)

        for i, name in enumerate(self.parameters):
            self.simulation1.set_constant(name, parameters[i])
            self.simulation2.set_constant(name, parameters[i])

        # Reset to ensure each simulate has same init condition
        self.simulation1.reset()
        self.simulation2.reset()
        self.simulation1.set_state(self.init_state)
        self.simulation2.set_state(self.init_state)

        # Run!
        try:
            self.simulation1.pre(100e3)
            self.simulation2.set_state(self.simulation1.state())
            p = Timeout(self.max_evaluation_time)
            d = self.simulation2.run(np.max(times)+0.02e3,
                log_times = times,
                log = [self._readout] + extra_log,
                #log_interval = 0.025
                progress=p,
                ).npview()
            del(p)
        except (myokit.SimulationError, myokit.SimulationCancelledError):
            return np.full(times.shape, float('inf'))

        # Apply capacitance filter and return
        if self.useFilterCap:
            d[self._readout] = d[self._readout] * self.fcap(times)

        if len(extra_log) > 0:
            return d
        return d[self._readout]

    def voltage(self, times, parameters=None):
        # Return voltage protocol

        # Update fix parameters
        self._set_fix_parameters(self._fix_parameters)

        if parameters is not None:
            # Update model parameters
            if self.transform is not None:
                parameters = self.transform(parameters)

            for i, name in enumerate(self.parameters):
                self.simulation1.set_constant(name, parameters[i])
                self.simulation2.set_constant(name, parameters[i])

        # Run
        self.simulation1.reset()
        self.simulation2.reset()
        self.simulation1.set_state(self.init_state)
        self.simulation2.set_state(self.init_state)
        try:
            self.simulation1.pre(100e3)
            self.simulation2.set_state(self.simulation1.state())
            d = self.simulation2.run(np.max(times)+0.02e3,
                log_times = times,
                log = ['membrane.V'],
                #log_interval = 0.025
                ).npview()
        except myokit.SimulationError:
            return float('inf')
        # Return
        return d['membrane.V'] 

    def EK(self):
        return self._EK
    
    def parameter(self):
        # return the name of the parameters
        return self.parameters

    def fix_parameters(self):
        # return fixed parameters and values
        return self._fix_parameters

#
# Model with leak
#
class ModelWithLeak(Model):
    def __init__(self, model_file, protocol_def, temperature,
                 transform=None, useFilterCap=False, effEK=True,
                 readout='voltageclamp.Iin', max_evaluation_time=5, leak=None):
        super(ModelWithLeak, self).__init__(model_file, protocol_def,
                temperature, transform=transform, useFilterCap=useFilterCap,
                effEK=effEK, readout=readout,
                max_evaluation_time=max_evaluation_time)
        self.set_leak(leak)
        self.set_delta(None)

    def set_leak(self, leak):
        # Set leak trace for fitting
        self._leak = leak

    def n_parameters(self):
        # n_parameters() method for Pints
        if self.delta is None:
            return len(self.parameters) + 1  # one extra parameter for leak
        else:
            return len(self.parameters)

    def set_delta(self, delta=None):
        self.delta = delta

    def simulate(self, parameters, times, extra_log=[]):
        # simulate() method for Pints

        if self.delta is None:
            # Last one is leak scaling
            delta = parameters[-1]
            parameters = parameters[:-1]
        else:
            # leak scaling has been set
            delta = self.delta
            parameters = np.copy(parameters)

        # Update fix parameters
        self._set_fix_parameters(self._fix_parameters)

        # Update model parameters
        if self.transform is not None:
            parameters = self.transform(parameters)

        for i, name in enumerate(self.parameters):
            self.simulation1.set_constant(name, parameters[i])
            self.simulation2.set_constant(name, parameters[i])

        # Reset to ensure each simulate has same init condition
        self.simulation1.reset()
        self.simulation2.reset()
        self.simulation1.set_state(self.init_state)
        self.simulation2.set_state(self.init_state)

        # Run!
        try:
            self.simulation1.pre(100e3)
            self.simulation2.set_state(self.simulation1.state())
            p = Timeout(self.max_evaluation_time)
            d = self.simulation2.run(np.max(times)+0.02e3,
                log_times = times,
                log = [self._readout] + extra_log,
                #log_interval = 0.025
                progress=p,
                ).npview()
            del(p)
        except (myokit.SimulationError, myokit.SimulationCancelledError):
            return np.full(times.shape, float('inf'))

        # Apply capacitance filter and return
        if self.useFilterCap:
            d[self._readout] = d[self._readout] * self.fcap(times)

        # Add leak
        d[self._readout] = d[self._readout] + self._leak * delta

        if len(extra_log) > 0:
            return d
        return d[self._readout]

#
# Model with common parameters
#
class MultiCellModel(Model):

    def __init__(self, model_file, protocol_def, temperature, n_cells,
                 n_common_parameters, n_independent_parameters,
                 transform_common=None, transform_independent=None,
                 useFilterCap=False, effEK=True, readout='voltageclamp.Iin',
                 max_evaluation_time=5):

        super(MultiCellModel, self).__init__(model_file, protocol_def,
                temperature, transform=None, useFilterCap=useFilterCap,
                effEK=effEK, readout=readout,
                max_evaluation_time=max_evaluation_time)

        self._n_cells = n_cells
        self._n_cp = n_common_parameters
        self._n_ip = n_independent_parameters

        self._transform_cp = transform_common
        self._transform_ip = transform_independent

        self._fix_parameters = [{}] * self._n_cells

    def n_parameters(self):
        return self._n_cp + self._n_cells * self._n_ip

    def set_fix_parameters(self, parameters):
        # Set/update parameters to a fixed value
        if len(parameters) != self._n_cells:
            raise ValueError('Input fix parameters must have the same length'
                             ' as `n_cells`.')
        self._fix_parameters = parameters

    def simulate(self, parameters, times, extra_log=[]):
        # simulate() method for Pints

        # Update common parameters (all cells shares)
        if self._transform_cp is not None:
            # TODO, here missing one conductance in the transform,
            # but should be OK for now...
            cp = self._transform_cp(parameters[:self._n_cp])

        for i, name in enumerate(self.parameters[1:self._n_cp + 1]):
            # + 1 is for conductance
            self.simulation1.set_constant(name, cp[i])
            self.simulation2.set_constant(name, cp[i])

        output = np.zeros((len(times), self._n_cells))

        for i_sim in range(self._n_cells):
            # Update fix parameters (cell dependent)
            self._set_fix_parameters(self._fix_parameters[i_sim])

            # Update independent parameters (cell dependent ones)
            if self._transform_ip is not None:
                # TODO, a bit messy here, conductance uses the same
                # log-transform, so put all to the same transform_ip
                ip = self._transform_ip(parameters[
                        self._n_cp + i_sim * self._n_ip:
                        self._n_cp + (i_sim + 1) * self._n_ip])

            # Set conductance
            self.simulation1.set_constant(self.parameters[0], ip[0])
            self.simulation2.set_constant(self.parameters[0], ip[0])

            for i, name in enumerate(self.parameters[self._n_cp + 1:]):
                # + 1 is for conductnace
                self.simulation1.set_constant(name, ip[i + 1])
                self.simulation2.set_constant(name, ip[i + 1])

            # Reset to ensure each simulate has same init condition
            self.simulation1.reset()
            self.simulation2.reset()
            self.simulation1.set_state(self.init_state)
            self.simulation2.set_state(self.init_state)

            # Run!
            try:
                self.simulation1.pre(100e3)
                self.simulation2.set_state(self.simulation1.state())
                p = Timeout(self.max_evaluation_time)
                d = self.simulation2.run(np.max(times)+0.02e3,
                    log_times = times,
                    log = [self._readout] + extra_log,
                    #log_interval = 0.025
                    progress=p,
                    ).npview()
                del(p)
            except (myokit.SimulationError, myokit.SimulationCancelledError):
                return np.full(output.shape, float('inf'))

            # Apply capacitance filter and return
            if self.useFilterCap:
                d[self._readout] = d[self._readout] * self.fcap(times)

            output[:, i_sim] = d[self._readout]

        return output

    def voltage(self, times):
        raise NotImplementedError


import parallel_evaluation as pe
import multiprocessing

class ParallelMultiCellLogLikelihood(pints.LogPDF):

    def __init__(self, list_of_likelihoods, n_common_parameters,
            n_independent_parameters, n_workers=None):
        """
        A wrapper that runs a given ``list_of_likelihoods`` in parallel with
        ``n_workers``.

        ``n_common_parameters`` and ``n_independent_parameters`` are needed to
        split the parameters in order.
        """

        self._lls = list_of_likelihoods
        self._n_ll = len(self._lls)
        self._n_cp = n_common_parameters
        self._n_ip = n_independent_parameters

        # Determine number of workers
        self._set_n_workers(n_workers)
        print('\nNumber of processors (in use): ' + str(self._nworkers) + '\n')
        self._ll_in_parallel = pe.ParallelEvaluator(self._lls,
                nworkers=self._nworkers)

    def _set_n_workers(self, n=None):
        # Set number of workers running in parallel
        if n is None:
            n = max(1, multiprocessing.cpu_count() - 1)
            n = min(self._n_ll, n)
        self._nworkers = n

    def n_parameters(self):
        return self._n_cp + self._n_ll * self._n_ip

    def _get_splitted_parameters(self, x):
        # Return a splitted up parameters from x that consists of common and
        # independent parameters
        splitted_parameters = []
        ncp = self._n_cp
        nip = self._n_ip
        cp = x[:ncp]

        for i in range(self._n_ll):
            ix = np.append(cp[:], x[ncp + i * nip:ncp + (i + 1) * nip])
            splitted_parameters.append(ix)

        return splitted_parameters

    def __call__(self, x):
        # __call__() method for Pints
        splitted_parameters = self._get_splitted_parameters(x)

        ff = self._ll_in_parallel.evaluate(splitted_parameters)

        tll = 0
        for ll in ff:
            if np.isnan(ll) or np.isinf(ll):
                ll = -np.inf
            tll += ll

        return tll

    def problem_evaluate(self, x):
        # Run problem.evaulate() within each individual log likelihood function
        splitted_parameters = self._get_splitted_parameters(x)

        output = np.zeros((self._lls[0]._nt, self._n_ll))
        for i, (ll, sp) in enumerate(zip(self._lls, splitted_parameters)):
            output[:, i] = ll._problem.evaluate(sp)

        return output


#
# Scheme 3 fit same kinetics
#
import multiprocessing.sharedctypes
class ParallelMultiLevelLogLikelihood(pints.LogPDF):

    def __init__(self,
            total_common_logposterior,
            list_of_independent_logposteriors,
            common_param,
            independent_param,
            transform_to_common_param,
            transform_to_independent_param,
            transformed_independent_parameters0,
            n_opt,
            restart_fit=True,
            onlyNelderMead=False,
            n_workers=None):
        """
        A wrapper that runs a given ``list_of_likelihoods`` in parallel.

        ``n_common_parameters`` and ``n_independent_parameters`` are needed to
        split the parameters in order.
        """

        self._clp = total_common_logposterior
        self._ilps = list_of_independent_logposteriors
        self.transform_to_common_param = transform_to_common_param
        self.transform_to_independent_param = transform_to_independent_param
        self.tip0 = transformed_independent_parameters0
        self.common_param = common_param
        self.independent_param = independent_param
        self.n_opt = n_opt
        self.onlyNelderMead = onlyNelderMead

        self._n_ll = len(self._ilps)
        self._n_cp = len(self.common_param)
        self._n_ip = len(self.independent_param)

        self.current_cp = np.zeros(self._n_cp)
        self.current_ip = np.zeros((self._n_ll, self._n_ip))
        self.best_tip = np.copy(transformed_independent_parameters0)
        self.best_lp = -1 * np.inf * np.ones(self._n_ll)

        self.restart_fit = restart_fit

        # Determine number of workers
        self._set_n_workers(n_workers)
        print('\nNumber of processors (in use): ' + str(self._nworkers) + '\n')
        '''
        self.current_tip_tmp = multiprocessing.Manager().dict()  # shared memory
        self.current_lp_tmp = multiprocessing.Manager().dict()  # shared memory
        args = [self.current_tip_tmp, self.current_lp_tmp, self.restart_fit]
        self._parallel_evaluator = pints.ParallelEvaluator(self._func,
                n_workers=self._nworkers, args=args)
        '''

    def _set_n_workers(self, n=None):
        # Set number of workers running in parallel
        if n is None:
            n = max(1, multiprocessing.cpu_count() - 1)
            n = min(self._n_ll, n)
        self._nworkers = n

    def n_parameters(self):
        return self._n_cp

    def get_fix_param(self, var, val):
        """
            var: variable name.
            val: variable value to fix.
        """
        out = {}
        for i, j in zip(var, val):
            out[i] = j
        return out

    def update_logposterior_fix_param(self, l, p):
        """
            l: log-posterior.
            p: dict, fix params.

            NOTE: Only works for specific implementation.
        """
        if 'delta' in p.keys():
            l._log_likelihood._problem._model.set_delta(p['delta'])
            tmp_p = {x: p[x] for x in p if x not in ['delta']}
        else:
            tmp_p = p
        l._log_likelihood._problem._model.set_fix_parameters(tmp_p)

    def _func(self, i, current_tip, current_lp, restart_x0):
        # Optimise the `i`th voltage artefact model and update the current
        # best independent parameters `self._best_ip[i]`.

        # Update model with current common parameters (from __call__())
        fix_cp = self.get_fix_param(self.common_param, self.current_cp)
        #print(self.current_cp, fix_cp)
        self.update_logposterior_fix_param(self._ilps[i], fix_cp)

        epsilon = 0  # for only Nelder-Mead

        # x0 uses the last value.
        if restart_x0:
            x0 = self.tip0[i]
        else:
            x0 = self.best_tip[i]

        if not self.onlyNelderMead:
            epsilon = 0.5  # when to switch from CMA-ES to Nelder-Mead
            # Create optimiser part 1
            opt = pints.OptimisationController(
                    self._ilps[i], x0,
                    method=pints.CMAES)
            opt.set_max_iterations(int(epsilon * self.n_opt))
            opt.set_log_to_screen(False)
            opt.set_parallel(False)

            # Run optimisation
            try:
                with np.errstate(all='ignore'):
                    # Tell numpy not to issue warnings
                    p, s = opt.run()
            except ValueError:
                import traceback
                traceback.print_exc()

            del(opt)
            #print(i, self.transform_to_independent_param(p))
            x0 = np.copy(p)

        # Create optimiser part 2
        opt = pints.OptimisationController(
                self._ilps[i], x0,
                method=pints.NelderMead)
        opt.set_max_iterations(int((1 - epsilon) * self.n_opt))
        opt.set_log_to_screen(False)
        opt.set_parallel(False)

        # Run optimisation
        try:
            with np.errstate(all='ignore'):
                # Tell numpy not to issue warnings
                p, s = opt.run()
        except ValueError:
            import traceback
            traceback.print_exc()

        # Update current independent parameters
        current_tip[i] = np.copy(p)
        current_lp[i] = np.copy(s)
        del(opt, p, s)

    def __call__(self, x):
        # __call__() method for Pints

        # Update the current common parameters
        self.current_cp = self.transform_to_common_param(x)
        #print('x: ', self.transform_to_common_param(x))

        '''
        self._parallel_evaluator.evaluate(range(self._n_ll))
        current_tip_tmp = self.current_tip_tmp
        current_lp_tmp = self.current_lp_tmp
        #'''
        #'''
        current_tip_tmp = multiprocessing.Manager().dict()  # shared memory
        current_lp_tmp = multiprocessing.Manager().dict()  # shared memory
        #'''
        '''
        # Create input args
        input_args = [[i, current_tip_tmp, current_lp_tmp, self.restart_fit]
                for i in range(self._n_ll)]
        scoop_futures.map(self._func, input_args)
        #'''
        '''
        pool = multiprocessing.Pool(processes=self._nworkers)
        # Create input args
        input_args = [[i, current_tip_tmp, current_lp_tmp, self.restart_fit]
                for i in range(self._n_ll)]
        pool.map(self._func, input_args)
        pool.close()
        pool.join()
        del(pool)
        #'''
        #'''
        procs = []  # TODO Add Queue() to limit nworkers?
        for i in range(self._n_ll):
            p = multiprocessing.Process(target=self._func,
                    args=(i, current_tip_tmp, current_lp_tmp,
                        self.restart_fit))
            procs.append(p)
            p.start()

        for p in procs:
            p.join()
        #'''
        '''
        for i in range(self._n_ll):
            self._func(i, current_tip_tmp, current_lp_tmp)
        #'''

        # Update the current independent parameters and calculate total lp
        total_lp = 0
        for i in range(self._n_ll):
            tip = np.copy(current_tip_tmp[i])
            self.current_ip[i][:] = self.transform_to_independent_param(tip)
            total_lp += current_lp_tmp[i]
            if current_lp_tmp[i] > self.best_lp[i]:
                # update x0 (best_tip) only with a better lp in case it runs
                # into weird parameter space...
                self.best_tip[i][:] = np.copy(tip)
                self.best_lp[i] = current_lp_tmp[i]

        del(current_tip_tmp, current_lp_tmp, procs)

        return total_lp

        '''
        # Update model with the 'best' independent parameters
        for i, lpcp in enumerate(self._clp._log_likelihoods):
            fix_ip = self.get_fix_param(self.independent_param,
                self.current_ip[i])
            #print(i, fix_ip)
            self.update_logposterior_fix_param(lpcp, fix_ip)

        return self._clp(x)
        '''

    def get_current_independent_param(self):
        return np.copy(self.current_ip)

    def problem_evaluate(self, x, ips):
        # Run problem.evaulate() within each individual log likelihood function
        output = np.zeros((self._ilps[0]._log_likelihood._nt, self._n_ll))

        for i, (lpcp, ip) in enumerate(zip(self._clp._log_likelihoods, ips)):
            fix_ip = self.get_fix_param(self.independent_param, ip)
            self.update_logposterior_fix_param(lpcp, fix_ip)
            output[:, i] = lpcp._log_likelihood._problem.evaluate(x)

        return output


#
# Setup own log-likelihood
#
class KnownNoiseLogLikelihood(pints.LogPDF):
    """
    Self define log-likelihood for multi-time series problem with some common
    parameters
    """
    def __init__(self, model, times, values, sigma):
        super(KnownNoiseLogLikelihood, self).__init__()

        # Store counts
        self._no = model._n_cells
        self._np = model.n_parameters()
        self._nt = len(times)

        self._model = model
        self._values = values
        self._times = times

        # Check sigma
        sigma = pints.vector(sigma)
        if len(sigma) != self._no:
            raise ValueError('Sigma must be a vector of length n_outputs.')
        if np.any(sigma <= 0):
            raise ValueError('Standard deviation must be greater than zero.')

        # Pre-calculate parts
        self._offset = -0.5 * self._nt * np.log(2 * np.pi)
        self._offset -= self._nt * np.log(sigma)
        self._multip = -1 / (2.0 * sigma**2)

    def n_parameters(self):
        return self._np

    def __call__(self, x):
        error = self._values - self._model.simulate(x, self._times)
        return np.sum(self._offset + self._multip * np.sum(error**2, axis=0))

