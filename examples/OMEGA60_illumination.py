from pydart import initialise_simulation, load_config

# First get the simulation configuration from the TOML input deck
config = load_config("configs/simulations/OMEGA60_500um.toml")

# Then initialise the simulation using the configuration
# This creates the simulation objects like target, laser, beams, etc. and sets up the simulation environment
simulation = initialise_simulation(config)

# Now you can run the simulation using the initialised objects and configuration
results = simulation.run()

# Calculate global and spherical-harmonic metrics.
metrics = results.get_metrics()

# You can also save the results to the output directory specified in the configuration
if config.simulation.save_deposition_data:
    results.save_deposition_data(config.simulation.output_directory)
if config.simulation.save_metrics:
    metrics.save_to_directory(config.simulation.output_directory)
if config.simulation.plot_data:
    from pydart.plotting import save_key_plots

    plot_path = save_key_plots(
        results,
        metrics,
        config.simulation.output_directory,
        dpi=config.simulation.plot_dpi,
    )
    print(f"Saved key plots to {plot_path}")

print(f"Deposited fraction: {float(metrics.deposited_fraction):.6f}")
print(f"RMS nonuniformity: {float(metrics.rms_nonuniformity):.6e}")
