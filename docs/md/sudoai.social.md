# Social

## Module Content

Social module
@author: Aymen Jemi (jemix) <[jemiaymen@gmail.com](mailto:jemiaymen@gmail.com)> at SUDO-AI

The social module is for extract data from social media.

### Examples

These examples illustrate how to use `Instagram` class.

```python
>>> insta = Instagram(username='foulane', password='1234abcd',max=20)
>>> comments = insta.get_comments('CRc3ZTGjxHr')
>>> insta.username2fullname('jemiaymen')
'أيمن الجامي'
```


### class sudoai.social.Instagram(token: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None, max: int = 12, verbose: int = 0)
Bases: `object`

Extract data from instagram posts.


#### verbose()
Verbose value.


* **Type**

    int



#### access_token()
Auth Token for authentication.


* **Type**

    str



#### session()
Current session.


* **Type**

    `requests.Session`



#### query_tasks()
Tasks like user_post,comment …


* **Type**

    dict



#### max()
Maximum comment by page.


* **Type**

    int



#### comment()
All comments.


* **Type**

    dict



#### username()
Current username.


* **Type**

    str



#### password()
Current password.


* **Type**

    str



#### is_auth()
If True current class is authenticated.


* **Type**

    bool



#### user()
Current user details.


* **Type**

    dict



#### headers()
Current session headers.


* **Type**

    str



#### get_comments(shortcode: str, with_full_name: bool = False)
Get comments from instagram post with shortcode.


* **Parameters**

    
    * **shortcode** (*str*) – Short code from instagram post.


    * **with_full_name** (*bool**, **optional*) – If True get full name for comment user (take more time). Defaults to False.



* **Returns**

    Dict with count of comments and data.



* **Return type**

    dict



#### load_session()
Load saved session.


* **Raises**

    **Exception** – When username is None.



#### login()
Authentification logic.


* **Returns**

    If auth is valid True, else False.



* **Return type**

    bool



#### save_session()
Save current session.


* **Returns**

    If session saved True else False.



* **Return type**

    bool



#### username2fullname(username: str)
Convert username to fullname.


* **Parameters**

    **username** (*str*) – username to convert.



* **Returns**

    Fullname.
    None: If username not exist.



* **Return type**

    str



#### valid_session()
Test if saved session is a valid session.
