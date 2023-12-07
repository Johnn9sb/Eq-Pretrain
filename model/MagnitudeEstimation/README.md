# Github 學習筆記

## repository 遠端版本庫
* 查看遠端
```shell=
# 顯示名稱
$ git remote

# 顯示名稱與 URL
$ git remove -v 
```

* 新增新的 repo
    * ```fetch```: 只會下載尚未更改的資料，不會與本地端合併，之後還是要手動合併 
    (**pull** 在沒有衝突時會直接合併)
    * **pull** 可以想像成是 fetch + 手動合併的過程
```shell=
$ git remote add <remote_name> <url>

# 獲取遠端 repo 資料
$ git fetch <remote_name>

# fetch 完合併當潛在的 branch 與 <branch_name>
$ git merge <branch_name>
```

* 移除或重新命名
```shell=
# rename
$ git remote rename <remote_name> <new_remote_name>

# remove
$ git remote rm <remote_naem>
```

## Push
```shell=
$ git push <remote_name> <branch_name>
```

## Branch
* Git 有個 ```HEAD``` 指標指向你正在工作的 branch 上

* 查看目前有的 branch
    * **"*"** 表示目前 checkout 的分支
```shell=
$ git branch

# 查看各分支最後一次提交的記錄
$ git branch -v

# 查看哪些分支已經/尚未被合併到當前 branch
$ git branch --merged
$ git branch --no-merged
```

* 建立新的分支，但不會切換到該 branch
```shell=
$ git branch <branch_name>
```

* 查看目前 HEAD 指標指向何處
```shell=
$ git log --decorate
```

* 切換 HEAD 到指定 branch
```shell=
$ git checkout <branch_name>

# 同時做 create 與切換到指定 branch
$ git checkout -b <branch_name>
```

* 刪除沒用的 branch
```shell=
$ git branch -d <branch_name>
```

