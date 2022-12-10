import os
import shutil
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Iterable, Tuple, Optional, Sequence, List
from tqdm import tqdm
from git import Repo
from affinity_data.java_parser import add_methods_from_java_tree, parse_file_and_add_to_project
from affinity_data.data_representations import UnexpectedUnparseableSourceError, \
    DataScrape, ScrappedProject

# Setup multiprocessing
import multiprocessing

from affinity_data.similar_api_project_list import PPair
from affinity_data.similar_api_project_list import similar_api_pairs

num_procs = multiprocessing.cpu_count() * 0.25
if num_procs > 1:
    import ray
    ray.init(num_cpus=4)


def git_clone(repo_url, repo_path):
    Repo.clone_from(repo_url, repo_path)
    return


"""
This file parses all of the separate Java files inside the git repo.
"""


def look_up_java_files(repo_path) -> ScrappedProject:
    project_name = repo_path
    project_list, files_skipped = [], 0
    project = ScrappedProject(
        lang="java",
        project_name=project_name,
        classes=[],
        failed_classes=[],
        ignored_classes=[]
    )
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                class_name = file
                parse_file_and_add_to_project(
                    class_name, file_path, project, print_failing_files=False)
    return project


def clone_and_parse_project(project: 'PotentialProject') -> ScrappedProject:
    #print(f"Loading {project.repo_path}")
    repo_url = f"https://github.com/{project.repo_path}.git"
    safe_repo_path = Path("./data/cloned_repos") / project.repo_path
    safe_repo_path.parent.mkdir(exist_ok=True, parents=True)
    try:
        git_clone(repo_url, str(safe_repo_path))
        scrapped_project = look_up_java_files(safe_repo_path)
    finally:
        # Be careful here! It is deleting files. #
        shutil.rmtree(str(safe_repo_path), ignore_errors=True)
    print(f"Parsed {project.repo_path} with {len(scrapped_project.classes)} classes and "
          f"{len(scrapped_project.failed_classes)} failed classes")
    return scrapped_project


def ray_pmap(func, items, desc: str = None, procs=num_procs):
    """Runs a multiprocess map using ray with tqdm support"""
    if procs >= 1:
        remote_func = ray.remote(func)
        futures = [remote_func.remote(item) for item in items]
        # https://github.com/ray-project/ray/issues/5554
        def to_iterator(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])
        return list(tqdm(to_iterator(futures), total=len(futures), desc=desc))
    else:
        items = list(items)
        return list(tqdm(map(func, items), total=len(items), desc=desc))


def main(
    projects: Iterable[Union['PotentialProject', 'PPair']],
    out_path: str = "affinity_jsonparse.pkl"
):
    """
	This function clones the repos into 'cloned_repos', and runs the parsing code.
	"""
    shutil.rmtree("./data/cloned_repos", ignore_errors=True)
    proj_objs, proj_pair_inds = _get_projs_and_inds_from_inputs(projects)
    desc = "Parsing files in projects"
    all_projects = ray_pmap(clone_and_parse_project, proj_objs, desc=desc)
    data_scrape = DataScrape(all_projects, proj_pair_inds)
    print("Writing to file...")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_scrape.serialize(out_path.expanduser())
    print("Success!")


def _get_projs_and_inds_from_inputs(
    projects: Iterable[Union['PotentialProject', 'PPair']]
) -> Tuple[List['PotentialProject'], List[Tuple[int, int]]]:
    outputted_projs_inds = {}
    proj_objs = []
    pair_inds = []

    def append_proj(p: PotentialProject):
        if p.repo_path in outputted_projs_inds:
            return  # Already have this project
        outputted_projs_inds[p.repo_path] = len(proj_objs)
        proj_objs.append(p)

    for p in projects:
        if isinstance(p, PotentialProject):
            append_proj(p)
        elif isinstance(p, PPair):
            projs = _convert_pair_to_potential_projects(p)
            for proj in projs:
                append_proj(proj)
            pair_inds.append((
                outputted_projs_inds[projs[0].repo_path],
                outputted_projs_inds[projs[1].repo_path],
            ))
        else:
            raise ValueError

    return proj_objs, pair_inds


@dataclass
class PotentialProject:
    repo_path: str
    description: str


def parse_project_list_file(
    path: Union[Path, str],
    #allowed_descriptions: Tuple[str, ...] = ("java",)
) -> Iterable[PotentialProject]:
    if isinstance(path, str):
        path = Path(path)

    for line in path.read_text(encoding='cp1251').split("\n"):
        fields = line.split("\t")
        repo_path = fields[0]
        description = fields[2] if len(fields) >= 2 else None
        #if not any(good_desc in description.lower() for good_desc in allowed_descriptions):
        #    continue
        yield PotentialProject(repo_path, description)


def _convert_pair_to_potential_projects(
    pair: PPair
) -> Tuple['PotentialProject', 'PotentialProject']:
    return (
        PotentialProject(pair.proj1, None),
        PotentialProject(pair.proj2, None),
    )



def run_from_hard_code_file(
    project_sample_size: Optional[int] = None,  # Randomly chose a count from the file
    include_first_n: int = 1000  # Only consider the first N from the file
):
    cur_file = Path(__file__).parent.absolute()
    projects_list_file = cur_file / "projects.txt"
    projects = list(parse_project_list_file(projects_list_file))
    if include_first_n:
        projects = projects[:min(len(projects), include_first_n)]
    if project_sample_size:
        projects = random.sample(projects, k=min(len(projects), project_sample_size))
    cur_file = Path(__file__).parent.absolute()
    main(
        projects=projects,
        out_path=str(
            cur_file / Path(f"../data/affinity-data/affinity_fromfile{len(projects)}-new.pkl.bz2")
        )
    )


def run_hard_code_projects():
    main(
        [
            PotentialProject("FasterXML/jackson-core", "java"),
            PotentialProject("google/gson", "java"),
        ],
        out_path="affinity_jsonparse.pkl.bz2"
    )


def run_from_similar_apis(
    pairs_sample_size: Optional[int] = None,  # Randomly chose n pairs
):
    pairs = similar_api_pairs
    if pairs_sample_size:
        pairs = random.sample(pairs, k=min(len(similar_api_pairs), pairs_sample_size))
    main(
        projects=pairs,
        out_path=f"~/data/new-affinity-data/affinity_frompairs{len(pairs)}.pkl.bz2"
    )


if __name__ == "__main__":
    run_from_hard_code_file()
    #run_from_similar_apis()
    #run_hard_code_projects()
