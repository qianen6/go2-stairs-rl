"""Complete deterministic development-report fixture, independent of production validation."""
def flat_report(passes=False):
    cases=[]
    for i in range(20):
        speed=.1 if i%2==0 else .2
        cases.append(dict(case_id=f'flat-dev-{i:02}',passed=passes,failure=None,complete=True,steps=900,
                          move_rmse_mps=0. if passes else speed,turn_rmse_radps=0. if passes else .2,
                          forward_distance_m=speed*10 if passes else 0.,commanded_distance_m=speed*10,
                          stable_stop_s=4.,longest_stop_s=4.))
    return dict(status='FLAT_DEVELOPMENT_EVALUATED',formal_stair_acceptance=False,scene_seed=2000,noise=False,
                checkpoint='fixture.pt',checkpoint_sha256='a'*64,cases=cases,passed_count=20 if passes else 0,passed=passes,
                protocol='18s: stand1, forward10(vx=.1/.2), turn3(yaw=.2), stop4; 20 parallel trials',
                thresholds=dict(no_termination=True,stable_stop_s=1.,move_rmse_mps=.1,
                                turn_rmse_radps=.15,minimum_commanded_distance_fraction=.5),
                action_metrics=dict(hard_clip_fraction=0.,near_bound_fraction=0.,max_abs_action=.05))
