from hypothesis import HealthCheck, settings, Verbosity

settings.register_profile(
    "default",
    verbosity=Verbosity.verbose,
    max_examples=5,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
settings.load_profile("default")
