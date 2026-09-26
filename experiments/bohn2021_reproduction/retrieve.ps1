$ErrorActionPreference = 'Stop'
$out = 'research_artifacts/bohn2021_reproduction_2026-09-17/search'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$requests = [ordered]@{
  author_repos = 'https://api.github.com/users/eivindeb/repos?per_page=100'
  horizon_title = 'https://api.github.com/search/repositories?q=%22prediction+horizon%22+reinforcement&per_page=100'
  paper_id = 'https://api.github.com/search/repositories?q=2102.11122&per_page=100'
  lmpc_horizon = 'https://api.github.com/search/repositories?q=lmpc-horizon&per_page=100'
  bohn_mpc = 'https://api.github.com/search/repositories?q=bohn+MPC&per_page=100'
  rlmpc_branches = 'https://api.github.com/repos/eivindeb/rlmpcopt/branches?per_page=100'
  rlmpc_tags = 'https://api.github.com/repos/eivindeb/rlmpcopt/tags?per_page=100'
  rlmpc_forks = 'https://api.github.com/repos/eivindeb/rlmpcopt/forks?per_page=100'
  gym_branches = 'https://api.github.com/repos/eivindeb/gym-letMPC/branches?per_page=100'
  gym_tags = 'https://api.github.com/repos/eivindeb/gym-letMPC/tags?per_page=100'
  gym_forks = 'https://api.github.com/repos/eivindeb/gym-letMPC/forks?per_page=100'
  original_repo = 'https://api.github.com/repos/eivindeb/lmpc-horizon'
  original_underscore = 'https://api.github.com/repos/eivindeb/lmpc_horizon'
  rlmpc_releases = 'https://api.github.com/repos/eivindeb/rlmpcopt/releases?per_page=100'
  gym_releases = 'https://api.github.com/repos/eivindeb/gym-letMPC/releases?per_page=100'
  zenodo = 'https://zenodo.org/api/records?q=%222102.11122%22&size=10'
}
$manifest = @()
foreach ($key in $requests.Keys) {
  $url = $requests[$key]
  try {
    $r = Invoke-RestMethod -Uri $url -TimeoutSec 30
    $r | ConvertTo-Json -Depth 40 | Set-Content -Encoding utf8 "$out/$key.json"
    $manifest += @{query=$key;url=$url;status='ok';at=(Get-Date).ToUniversalTime().ToString('o')}
    if ($key -in @('author_repos','rlmpc_forks','gym_forks')) { "$key : $(@($r).Count) records" }
    elseif ($r.PSObject.Properties.Name -contains 'total_count') { "$key : $($r.total_count) matches"; $r.items | Select-Object full_name,description | ConvertTo-Json -Depth 3 }
    else { "$key : saved" }
  } catch {
    $manifest += @{query=$key;url=$url;status='failed';error=$_.Exception.Message;at=(Get-Date).ToUniversalTime().ToString('o')}
    "$key : $($_.Exception.Message)"
  }
  $manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 "$out/manifest.json"
  Start-Sleep -Milliseconds 2200
}
